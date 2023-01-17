from collections import namedtuple

import sys

import lightgbm as lg

from tqdm import tqdm

import numpy as np

from sortedcontainers import SortedDict

from sklearn.metrics import roc_auc_score



EventBody = namedtuple('EventBody', [

                       'time', 'id', 'action', 'type', 'side', 'price', 'amount', 'is_snapshot', 'Y'])



class Event(EventBody):

    @property

    def need_prediction(self):

        return self.Y >= 0
class Side:

    BID = 0

    ASK = 1



class Action:

    DELETE = 0

    ADD = 1

    MODIFY_AMOUNT = 2

    DEAL = 3

    NEW_CHUNK = 10



MIN_PRICE = 0

MAX_PRICE = 10**10



DEBUG = False



class Order:

    def __init__(self, event):

        self.order_id = event.id

        self.side = event.side

        self.price = event.price

        self.type = event.type

        self.amount = event.amount

        self.initial_amount = event.amount

        self.time = event.time





class PriceLevel:

    def __init__(self, side, price):

        self.side = side

        self.price = price

        self.orders = []

        self.num_orders = 0

        self.total_amount = 0

        self.id_to_order = {}



    def add_order(self, order):

        self.orders.append(order)

        self.num_orders += 1

        self.total_amount += order.amount

        self.id_to_order[order.order_id] = order

        self._check_level()



    def _get_order(self, order_id):

        return self.id_to_order[order_id]



    def delete_order(self, order_id):

        order = self._get_order(order_id)



        self.num_orders -= 1

        self.total_amount -= order.amount

        self.orders.remove(order)

        del self.id_to_order[order_id]

        self._check_level()



    def modify_order(self, order_id, new_amount):

        order = self._get_order(order_id)

        change_amount = order.amount - new_amount

        assert change_amount > 0

        order.amount -= change_amount

        self.total_amount -= change_amount

        self._check_level()



    def get_volume(self):

        return self.total_amount



    def get_num_orders(self):

        return self.num_orders



    def get_orders(self):

        return self.orders



    def _check_level(self):

        if not DEBUG:

            return

        assert self.num_orders == len(self.orders)

        assert sum([o.amount for o in self.orders]) == self.total_amount

        





class OrderBook:

    def __init__(self):

        self._clear()



    def _clear(self):

        self.best_price = [MIN_PRICE, MAX_PRICE]

        self.price_levels = [SortedDict(), SortedDict()]

        self.time = 0

        self.events = []



    def _get_empty_price(self, side):

        return MIN_PRICE if side == Side.BID else MAX_PRICE



    def _get_updated_best_price(self, side):

        if len(self.price_levels[side]) == 0:

            return self._get_empty_price(side)

        if side == Side.BID:

            return self.price_levels[side].peekitem(-1)[0]  # max_key

        return self.price_levels[side].peekitem(0)[0]  # min_key



    def get_price_level(self, side, price):

        if price not in self.price_levels[side]:

            self.price_levels[side][price] = PriceLevel(side, price)

        return self.price_levels[side][price]



    def add_order(self, order):

        side, price = order.side, order.price

        self.get_price_level(side, price).add_order(order)

        self.best_price[side] = self._get_updated_best_price(side)



    def _del_if_empty(self, side, price):

        if self.price_levels[side][price].get_num_orders() == 0:

            del self.price_levels[side][price]

            if price == self.get_best_price(side):

                self.best_price[side] = self._get_updated_best_price(side)



    def delete_order(self, side, price, order_id):

        self.price_levels[side][price].delete_order(order_id)

        self._del_if_empty(side, price)

        

    def modify_order(self, side, price, order_id, new_amount):

        self.price_levels[side][price].modify_order(order_id, new_amount)

        self._del_if_empty(side, price)



    def apply_event(self, event):

        self.time = event.time

        self.events.append(event)



        if event.action == Action.DELETE:  # deletion

            self.delete_order(event.side, event.price, event.id)

        elif event.action == Action.MODIFY_AMOUNT:  # modifying

            self.modify_order(event.side, event.price, event.id, event.amount)

        elif event.action == Action.ADD:  # adding new order

            self.add_order(Order(event=event))

        elif event.action == Action.NEW_CHUNK: 

            self._clear()

        else:

            pass

        assert self.best_price[1] > self.best_price[0]



    def get_best_price(self, side):

        return self.best_price[side]



    def get_mean_price(self):

        return (self.best_price[Side.BID] + self.best_price[Side.ASK]) / 2.0



    def get_time(self):

        return self.time



    def get_events(self):

        return self.events



    def get_price_at_ix(self, side, index):

        return self.best_price[side] - index * (1 - 2 * side)



    def get_ix_at_price(self, side, price):

        return (self.best_price[side] - price) * (1 - 2 * side)



    def get_price_level_at_ix(self, side, ix):

        price = self.get_price_at_ix(side, ix)

        return self.price_levels[side].get(price, None)





class EventPlayer:

    def __init__(self, filename):

        self.events = np.load(filename)["events"]



    def iget_events(self):

        for event in self.events:

            yield Event(*event)



    def __len__(self):

        return len(self.events)
SIDE_BID = 0 

SIDE_ASK = 1



def get_simple_features_from_orderbook(orderbook, max_index=2):

    '''

        Getting simple features from the orderbook

    '''

    spread = orderbook.get_best_price(SIDE_ASK) - orderbook.get_best_price(SIDE_BID)

    features = [spread]

    for side in (SIDE_BID, SIDE_ASK):

        for ix in range(max_index):

            price_level = orderbook.get_price_level_at_ix(side, ix)

            if price_level is None:

                features += [-1, -1]

            else:

                features += [price_level.get_volume(), 

                             price_level.get_num_orders()]

    return features





def get_simple_deals_features(last_deals, orderbook):

    '''

        Getting simple features from the last deals

    '''

    cur_mean_price = orderbook.get_mean_price()

    cur_time = orderbook.get_time()



    features = []

    for side in (SIDE_BID, SIDE_ASK):

        deal_event = last_deals[side]

        if deal_event is None:

            features += [-1e9, -1e9, -1e9]

        else:

            features += [cur_mean_price - deal_event.price, 

                         cur_time - deal_event.time, 

                         deal_event.amount]

    return features





def collect_dataset(data_path):

    '''

        Collecting dataset

    '''

    event_player = EventPlayer(data_path)

    orderbook = OrderBook()



    X = []

    Y = []



    last_deals = [None, None]

    for ev in tqdm(event_player.iget_events(), 

                    total=len(event_player), 

                    desc="collecting dataset"):

        if ev.action == Action.DEAL:

            last_deals[ev.side] = ev

        elif ev.action == Action.NEW_CHUNK:

            last_deals = [None, None]

        

        orderbook.apply_event(ev)

        if ev.need_prediction:

            features = get_simple_features_from_orderbook(orderbook)

            features += get_simple_deals_features(last_deals, orderbook)



            X.append(features)

            Y.append(ev.Y)



    print(f"Dataset collected: len(X) = {len(X)}")

    return np.array(X), np.array(Y)





X_train, Y_train = collect_dataset("../input/orderbooks-events-binary-classification/train_small_A.npz")

X_test, Y_test = collect_dataset("../input/orderbooks-events-binary-classification/train_small_B.npz")
def train_classifier(X_train, Y_train, X_test, Y_test):

    '''

        Classifier training

    '''

    clf = lg.LGBMClassifier(num_leaves=31, n_estimators=1000, learning_rate=0.1)

    clf.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], 

            eval_metric="auc", early_stopping_rounds=20)

    return clf

 

clf = train_classifier(X_train, Y_train, X_test, Y_test)
last_deals = [None, None]



def process_event_and_predict_proba(event, orderbook):

    if event.action == Action.DEAL:

        last_deals[event.side] = event

    elif event.action == Action.NEW_CHUNK:

        last_deals[:] = [None, None]

        

    if not event.need_prediction:

        return None

    

    features = get_simple_features_from_orderbook(orderbook)

    features += get_simple_deals_features(last_deals, orderbook)    

    proba = clf.predict_proba([features])[0, 1]

    return proba
class Scorer:

    def __init__(self, fname):

        self.fname = fname

        self.event_player = EventPlayer(fname)



    def _check_return_value(self, event, pred_proba):

        if event.need_prediction:

            if pred_proba is None:

                raise ValueError("You should return probability if event.need_prediction == True")

            if not (0 <= pred_proba <= 1):

                raise ValueError("Predicted probability is not in [0, 1] range")

        else:

            if pred_proba is not None:

                raise ValueError("Return probability should be None if event.need_prediction == False")



    def score(self, process_event_func):

        Ys = []

        pred_probas = []

        orderbook = OrderBook()



        for event in tqdm(self.event_player.iget_events(), 

                            total=len(self.event_player),

                            desc="scoring"):        

            orderbook.apply_event(event)

            pred_proba = process_event_func(event, orderbook)

            self._check_return_value(event, pred_proba)



            if not event.need_prediction:

                continue



            Ys.append(event.Y)

            pred_probas.append(pred_proba)



        roc_auc = roc_auc_score(Ys, pred_probas)

        print(f"\nroc_auc_score = {roc_auc:.3f}")

        return roc_auc, (Ys, pred_probas)
scoring = Scorer("../input/orderbooks-events-binary-classification/train_small_C.npz")

roc_auc, (true_ys, pred_probas) = scoring.score(process_event_and_predict_proba)