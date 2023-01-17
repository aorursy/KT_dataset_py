import os

for f in os.walk('../input/'):
    print(f)
import random
import string
import copy as cp

from datetime import date, datetime, timedelta


def read_graph_data(metadata_filename, graph_data_structure='up'):
    """ Read the user-review-product graph from file. Can output the graph in different formats

        Args:
            metadata_filename: a gzipped file containing the graph.
            graph_data_structure: defines the output graph format
                'up' (default) ---> user-product and product-user graphs
                'urp' ---> user-review and review-product graphs
        Return:
            graph1: user-product / user-review
            graph2: product-user / review-product
    """

    user_data = {}
    prod_data = {}

    min_date, max_date = None, None

    with open(metadata_filename, 'r') as f:
        # file format: each line is a tuple (user id, product id, rating, 1, date)
        for line in f:
            items = line.strip().split()
            u_id = items[0]
            p_id = items[1]
            rating = float(items[2])
            label = int(items[3])
            date = items[4]

            if u_id not in user_data:
                user_data[u_id] = []
            user_data[u_id].append((p_id, rating, label, date))

            if p_id not in prod_data:
                prod_data[p_id] = []
            prod_data[p_id].append((u_id, rating, label, date))

            cur_date = datetime.strptime(date, '%Y-%m-%d')
            if min_date is None or cur_date < min_date:
                min_date = cur_date
            if max_date is None or cur_date > max_date:
                max_date = cur_date
    print(min_date)
    print(max_date)

    if graph_data_structure == 'up':
        return user_data, prod_data

    if graph_data_structure == 'urp':
        user_review_graph = {}
        for k, v in user_data.items():
            user_review_graph[k] = []
            for t in v:
                user_review_graph[k].append((k, t[0]))  # (u_id, p_id) representing a review
        review_product_graph = {}
        for k, v in prod_data.items():
            for t in v:
                # (u_id, p_id) = (t[0], k) is the key of a review
                review_product_graph[(t[0], k)] = k
        return user_review_graph, review_product_graph


def write_graph_data(prod_user_graph, filename):
    """output a review graph to file"""
    with open(filename, 'w') as f:
        for k, v in prod_user_graph.items():
            for r in v:
                # user_id prod_id rating 1 date_str
                f.write(f'{r[0]} {k} {r[1]} 1 {r[3]}\n')
                

def evaluate_PE(user_product_graph, product_user_graph, target_products, elite_threshold, elite_accounts, print_=False):
    """
    This function calculates the Practical Effect (Revenues) of the target products on the given graph.

    :param user_product_graph: review graph in the user-centric format.
            Example: {user_id_1: [(product_id_1, rating, 1, date), (product_id_2, rating, 1, date)]}

    :param product_user_graph: review graph in the product-centric format.
            Example: {product_id_1: [(user_id_1, rating, 1, date), (user_id_2, rating, 1, date)]}

    :param target_products: the products whose revenues will be calculated.

    :param elite_threshold: the least number of reviews that an account must have to attain the elite status.
            If Not None, it is used to find elite users in ANY time.
            If None, elite status is defined by the next parameter.
    
    :param elite_accounts: a set of user ids that are considered elite.

    :param print_: if False, don't print intermediate calculations.
    
    :return: practical effect, defined as follows.
            For each target item, we calculate its
    """
    # calculate the original revenue
    avg_ratings = {}
    for product, reviews in product_user_graph.items():
        rating = 0
        if len(reviews) == 0:
            avg_ratings[product] = 0
        else:
            for review in reviews:
                rating += review[1]
            avg_ratings[product] = rating / len(reviews)
    mean_rating = sum([r for r in avg_ratings.values()]) / len(avg_ratings)

    RI = {}
    ERI = {}
    total_elite_reviews = 0
    for target in target_products:
        RI[target] = avg_ratings[target] - mean_rating
        temp_ERI = []
        for review in product_user_graph[target]:

            # if the user has attain elite status at the moment
            if elite_threshold is not None:
                if len(user_product_graph[review[0]]) >= elite_threshold:
                    temp_ERI.append(review[1])
            else:
                # use pre-defined elite status
                assert elite_accounts is not None
                if review[0] in elite_accounts:
                    temp_ERI.append(review[1])

        ERI[target] = sum(temp_ERI) / len(temp_ERI) if len(temp_ERI) != 0 else 0
        total_elite_reviews += len(temp_ERI)

    Revenue = {}
    for target in target_products:
        Revenue[target] = 0.09 + 0.35 * RI[target] + 0.01 * ERI[target]
        if print_:
            print(f'contribution of RI:ERF = {RI[target]}:{ERI[target]}')
    if print_:
        print(f'total elite reviews = {total_elite_reviews}')
    return RI, ERI, Revenue, total_elite_reviews


def attack(user_product_graph, num_spams, spammer_accounts, attack_period):
    """
    For each of the target items, generate a fixed number of spams.
    Here your algorithm needs to decide what users will post how many spams for a target item at what time and with what rating {1,2,3,4,5}.

    Constraints:
    1) ratings must be from {1,2,3,4,5};
    2) spam post time must be within the attack_period;
    3) the number of spams must be the same as specified in num_spams;
    4) the target products are specified by the keys of the dictionary num_spams.
    5) spmming account must be selected from spammer_accounts

    :param user_product_graph: review graph in the user-centric format.
            Example: {user_id_1: [(product_id_1, rating, 1, date), (product_id_2, rating, 1, date)]}

    :param num_spams: a dictionary in the form {item_id_1:num_spams_1, ..., item_id_n:num_spams_n}, where item_id_k is the id of a target product.

    :param spammer_accounts: a dictionary of user_ids and their elite status (1:elite, 0:non-elite) at the spammer's disposal.
            Example: {user_id_1: 1, user_id_2: 0} means that user_id_1 is an elite account and user_id_2 is not.

    :param time_span: [start_time, end_time]: all spams must have date time within this span

    :return: a list of reviews in the form of [(user_id_1, item_id_1, date_1, rating_1), ..., (user_id_n, item_id_n, date_n, rating_n)]
                new user_id can be registered and any user_id not in the existing graph will be considered as new accounts
                non-target items can also receive spams, but the Practical Effect will be calculated on the target items only.

            the spams in product_user_graph format for outputing the attack to a file to be submitted.
    """
    # a simple example: randomly pick an existing account to post spam for a single target item
    added_spams = []
    added_spams_prod_user_graph = {}

    start_date, end_date = attack_period
    day_span = end_date - start_date
    days_between_dates = day_span.days

    existing_user_ids = [k for k, _ in spammer_accounts.items()]

    user_prod_lookup_table = {k: set([r[0] for r in v]) for k, v in user_product_graph.items()}
    added_spams_lookup_table = set()

    # for each target product
    for item_id, number in num_spams.items():
        added_spams_prod_user_graph[item_id] = []
        # post this "number" of spams
        for i in range(number):
            selected_user_id = random.choice(existing_user_ids)

            # avoid users who have posted reviews for this item
            while item_id in user_prod_lookup_table[selected_user_id] \
                    or (selected_user_id, item_id) in added_spams_lookup_table:
                selected_user_id = random.choice(existing_user_ids)

            random_number_of_days = random.randrange(days_between_dates)
            random_date = start_date + timedelta(days=random_number_of_days)
            random_rating = random.choice([4, 5])
            date_time_format_str = '%Y-%m-%d'
            added_spams.append((selected_user_id, item_id, random_rating, random_date.strftime(date_time_format_str)))
            added_spams_prod_user_graph[item_id].append((selected_user_id, random_rating, 1, random_date.strftime(date_time_format_str)))
            added_spams_lookup_table.add((selected_user_id, item_id))
    return added_spams, added_spams_prod_user_graph


def add_spams_to_review_graph(user_product_graph, product_user_graph, spams):
    """
    This function add spams created by the spammers to the existing reviews.

    :param user_product_graph: review graph in the user-centric format.
            Example: {user_id_1: [(product_id_1, rating, 1, date), (product_id_2, rating, 1, date)]}

    :param product_user_graph: review graph in the product-centric format.
            Example: {product_id_1: [(user_id_1, rating, 1, date), (user_id_2, rating, 1, date)]}

    :param spams: the spams returned by the "attack" function.
                Format:
                    [(user_id_1, item_id_1, date_1, rating_1), ..., (user_id_n, item_id_n, date_n, rating_n)]

    Constraints:
    1) all spams' product_id are in the existing review data. Namely, the spammers can't create new products.
    2) a spam may have user_id that are not in the existing review data.

    :return: a copy of user_product_graph and product_user_graph with spams added.
    """
    user_product_graph_copy = cp.deepcopy(user_product_graph)
    product_user_graph_copy = cp.deepcopy(product_user_graph)

    for spam in spams:
        user_id, product_id, rating, date = spam
        if user_id not in user_product_graph_copy:
            user_product_graph_copy[user_id] = []
        user_product_graph_copy[user_id].append((product_id, rating, 1, date))

        assert product_id in product_user_graph_copy, 'Error: new product created by spammer.'

        product_user_graph_copy[product_id].append((user_id, rating, 1, date))

    return user_product_graph_copy, product_user_graph_copy


def remove_spams_from_review_graph(user_product_graph, product_user_graph, detected_spams):
    """
    This function removes the detected spams from a review graph.

    :param user_product_graph: review graph in the user-centric format.
            Example: {user_id_1: [(product_id_1, rating, 1, date), (product_id_2, rating, 1, date)]}

    :param product_user_graph: review graph in the product-centric format.
            Example: {product_id_1: [(user_id_1, rating, 1, date), (user_id_2, rating, 1, date)]}

    :param detected_spams: the spams detected by some detector.
                Format:
                    [(user_id_1, item_id_1), ..., (user_id_n, item_id_n)]

    Constraints:
    1) all spams' product_id and user_id are in the existing review data.

    :return: a copy of user_product_graph and product_user_graph with spams removed.
    """
    user_product_graph_copy = cp.deepcopy(user_product_graph)
    product_user_graph_copy = cp.deepcopy(product_user_graph)

    for spam in detected_spams:
        user_id, product_id = spam
        assert user_id in user_product_graph_copy, f'Error: user_id {user_id} not in review data.'
        prev_len = len(user_product_graph_copy[user_id])
        for review in user_product_graph_copy[user_id][:]:
            if review[0] == product_id:
                user_product_graph_copy[user_id].remove(review)
        last_len = len(user_product_graph_copy[user_id])
        assert prev_len - last_len >= 1, 'Error: can\'t remove a spam.'

        assert product_id in product_user_graph_copy, 'Error: new product created by spammer.'
        prev_len = len(product_user_graph_copy[product_id])
        for review in product_user_graph_copy[product_id][:]:
            if review[0] == user_id:
                product_user_graph_copy[product_id].remove(review)
        last_len = len(product_user_graph_copy[product_id])
        assert prev_len - last_len >= 1, 'Error: can\'t remove a spam.'

    return user_product_graph_copy, product_user_graph_copy
import math
from copy import deepcopy
from datetime import datetime
import numpy as np

class FeatureExtractor:
    date_time_format_str = '%Y-%m-%d'

    def __init__(self):
        # for the users, keep the normalizer of mnr
        self.user_mnr_normalizer = 0

        # for the products, keep the normalizer of mnr
        self.prod_mnr_normalizer = 0

        # keeping number of reviews and sum of ratings for each product
        self.product_num_ratings = {}
        self.product_sum_ratings = {}
        
    def MNR(self, data, data_type='user'):
        """
            Normalized maximum number of reviews in a day for a user/product
            Args:
                data is a dictionary with key=u_id or p_id and value = tuples of (neighbor id, rating, label, posting time)
            Return:
                dictionary with key = u_id or p_id and value = MNR
        """
        # maximum number of reviews written in a day for user / product
        feature = {}
        for i, d in data.items():
            # key = posting date; value = number of reviews
            frequency = {}
            for t in d:
                if t[3] not in frequency:
                    frequency[t[3]] = 1
                else:
                    frequency[t[3]] += 1
            feature[i] = max(frequency.values())

        # normalize it
        if data_type == 'user':
            self.user_mnr_normalizer = max(feature.values())
            for k in feature.keys():
                feature[k] /= self.user_mnr_normalizer
        else:
            self.prod_mnr_normalizer = max(feature.values())
            for k in feature.keys():
                feature[k] /= self.prod_mnr_normalizer

        return feature

    def RD(self, product_data):
        """Calculate the deviation of the review ratings to the product average.

            Args:
                prod_data:
            Return:
                a dictionary with key = (u_id, p_id), value = deviation of the rating of this review to the average rating of the target product
        """
        rd = {}
        for i, d in product_data.items():
            avg = np.mean(np.array([t[1] for t in d]))
            for t in d:
                rd[(t[0], i)] = abs(t[1] - avg)
        return rd
    
    def ISR(self, user_data):
        """
            Check if a user posts only one review
        """
        isr = {}
        for i, d in user_data.items():
            # go through all review of this user
            for t in d:
                if len(d) == 1:
                    isr[(i, t[0])] = 1
                else:
                    isr[(i, t[0])] = 0
        return isr

    def add_feature(self, existing_features, new_features, feature_names):
        """
            Add or update feature(s) of a set of nodes of the same type to the existing feature(s).
            If a feature of a node is already is existing_features, then the new values will replace the existing ones.
            Args:
                existing_features: a dictionary {node_id:dict{feature_name:feature_value}}
                new_features: new feature(s) to be added. A dict {node_id: list of feature values}
                feature_names: the name of the new feature. A list of feature names, in the same order of the list of feature values in new_features
        """

        for k, v in new_features.items():
            # k is the node id and v is the feature value
            if k not in existing_features:
                existing_features[k] = dict()
            # add the new feature to the dict of the node
            for i in range(len(feature_names)):
                if len(feature_names) > 1:
                    existing_features[k][feature_names[i]] = v[i]
                else:
                    existing_features[k][feature_names[i]] = v
                    
                    
    def construct_all_features(self, user_data, prod_data):
        """
            Main entry to feature construction.
            Args:
            Return:
                user, product and review features
        """

        # key = user id, value = dict of {feature_name: feature_value}
        UserFeatures = {}
        # key = product id, value = dict of {feature_name: feature_value}
        ProdFeatures = {}

        uf = self.MNR(user_data, data_type='user')
        self.add_feature(UserFeatures, uf, ["MNR"])
        
        pf = self.MNR(prod_data, data_type='prod')
        self.add_feature(ProdFeatures, pf, ["MNR"])

        ReviewFeatures = {}
        rf = self.RD(prod_data)
        self.add_feature(ReviewFeatures, rf, ['RD'])
        rf = self.ISR(user_data)
        self.add_feature(ReviewFeatures, rf, ['ISR'])

        return UserFeatures, ProdFeatures, ReviewFeatures
    
    
    def calculateNodePriors(self, feature_names, features_py, when_suspicious):
        """
            Calculate priors of nodes P(y=1|node) using node features.
            Args:
                feature_names: a list of feature names for a particular node type.
                features_py: a dictionary with key = node_id and value = dict of feature_name:feature_value
                when_suspicious: a dictionary with key = feature name and value = 'H' (the higher the more suspicious) or 'L' (the opposite)
            Return:
                A dictionary with key = node_id and value = S score (see the SpEagle paper for the definition)
        """
        priors = {}
        for node_id, v in features_py.items():
            priors[node_id] = 0

        for f_idx, fn in enumerate(feature_names):

            fv_py = []
            for node_id, v in features_py.items():
                if fn not in v:
                    fv_py.append((node_id, -1))
                else:
                    fv_py.append((node_id, v[fn]))
            fv_py = sorted(fv_py, key=lambda x: x[1])

            i = 0
            while i < len(fv_py):
                start = i
                end = i + 1
                while end < len(fv_py) and fv_py[start][1] == fv_py[end][1]:
                    end += 1
                i = end

                for j in range(start, end):
                    node_id = fv_py[j][0]
                    if fv_py[j][0] == -1:
                        priors[node_id] += pow(0.5, 2)
                        continue
                    if when_suspicious[fn] == '+1':
                        priors[node_id] += pow((1.0 - float(start + 1) / len(fv_py)), 2)
                    else:
                        priors[node_id] += pow(float(end) / len(fv_py), 2)

        for node_id, v in features_py.items():
            priors[node_id] = 1.0 - math.sqrt(priors[node_id] / len(feature_names))
            if priors[node_id] > 0.999:
                priors[node_id] = 0.999
            elif priors[node_id] < 0.001:
                priors[node_id] = 0.001
        return priors
    
    
def detectPrior(user_product_graph, prod_user_graph, product_range, spammer_account_range, top_k):
    """
    Detect spams on the input review graph.

    :param user_product_graph: input review graph
    :param prod_user_graph: input review graph
    :param product_range: within what products spams are to be detected
    :param spammer_account_range: within what users spams are to be detected
    :param top_k: how many reviews to return as spams.
    """
    UserPriors = {}
    ProdPriors = {}
    ReviewPriors = {}

    # feature configuration
    feature_config = {'MNR':'+1', 'ISR':'+1', 'RD':'+1'}

    review_feature_list = ['RD', 'ISR']
    user_feature_list = ['MNR']
    product_feature_list = ['MNR']

    extractor = FeatureExtractor()
    UserFeatures, ProdFeatures, ReviewFeatures = extractor.construct_all_features(user_product_graph, prod_user_graph)

    UserPriors = extractor.calculateNodePriors(user_feature_list, UserFeatures, feature_config)
    ProductPriors = extractor.calculateNodePriors(product_feature_list, ProdFeatures, feature_config)
    ReviewPriors = extractor.calculateNodePriors(review_feature_list, ReviewFeatures, feature_config)

    review_suspiciousness = []

    for prod_id, reviews in prod_user_graph.items():
        if prod_id not in product_range:
            continue
        for review in reviews:
            user_id, rating, _, date = review
            if user_id not in spammer_account_range:
                continue
            review_id = (user_id, prod_id)
            user_prior = UserPriors[user_id]
            prod_prior = ProductPriors[prod_id]
            review_prior = ReviewPriors[review_id]
            review_suspiciousness.append((review_id, np.log(user_prior) + np.log(prod_prior) + np.log(review_prior)))

    review_suspiciousness = sorted(review_suspiciousness, reverse=True, key=lambda x: x[1])
    return [r for r in review_suspiciousness[:top_k]]


def evaluation(user_product_graph,
               prod_user_graph,
               elite_users,
               product_range,
               spammer_account_range,
               target_products,
               spamming_accounts,
               attacking_spams,
               detector,
               top_k):
    """
    Evaluate a spam detection algorithm after attacking spams are added.

    Evaluation:
        1) compute the revenue of the target products without attacking spams
        2) add spams to the review data
        3) run the detector on the new data
        4) remove the detected spams from the graph
        5) recompute the revenue of the target products
        6) find the promoted revenue

    :param user_product_graph: input review graph
    :param prod_user_graph: input review graph
    :param elite_users: pre-defined set of elite users. Fixed throughout the attack and detection.
    :param product_range: within what products spams are to be detected
    :param spammer_account_range: within what users spams are to be detected
    :param target_products: the true target products and a subset of product_range
    :param spamming_accounts: the spammer accounts and a subset of spammer_account_range
    :param attacking_spams: spams added by the contestant algorithm. The spam data are downloaded from the submission site.
    :param detector: a detection algorithm, passed as the name of the function.
    :param top_k: the top k reviews with the highest suspiciousness scores will be regarded as positive.
    """

    # step 1: compute the revenue of the target products without attacking spams
    RI, ERI, Revenue, totalElite = evaluate_PE(user_product_graph, prod_user_graph, target_products, None,
                                               set(elite_users), print_=False)
    old_revenue = sum([Revenue[k] for k in Revenue.keys()])

    # step 2: add spams to the review data
    user_product_graph_copy, prod_user_graph_copy = add_spams_to_review_graph(user_product_graph, prod_user_graph,
                                                                              attacking_spams)
    RI, ERI, Revenue, totalElite = evaluate_PE(user_product_graph_copy, prod_user_graph_copy, target_products, None,
                                               set(elite_users), print_=False)

    # step 3: run the detector on the new data. The detection is constrained to product_range x spammer_account_range.
    predicted_spams = detector(user_product_graph_copy, prod_user_graph_copy, product_range, spammer_account_range, top_k)
    
    # remember to filter out the false positives.
    attacking_spams_lookup = set([(s[0], s[1]) for s in attacking_spams])
    predicted_spams_lookup = set([ds[0] for ds in predicted_spams])
    detected_spams = list(attacking_spams_lookup.intersection(predicted_spams_lookup))
    
    print(f'Precision: {float(len(detected_spams))/len(predicted_spams_lookup)}')
    print(f'Recall: {float(len(detected_spams))/len(attacking_spams)}')

    # step 4: remove the detected spams from the graph
    user_product_graph_copy_copy, prod_user_graph_copy_copy = remove_spams_from_review_graph(user_product_graph_copy,
                                                                                             prod_user_graph_copy,
                                                                                             detected_spams)

    # step 5: recompute the revenue of the target products
    newRI, newERI, newRevenue, newtotalElite = evaluate_PE(user_product_graph_copy_copy, prod_user_graph_copy_copy,
                                                           target_products, None, set(elite_users), print_=False)

    # step 6: find the difference in the target revenue
    print(f'Practical Effect Increase = {sum([newRevenue[k] for k in newRevenue.keys()]) - old_revenue}')
import pickle
import pandas as pd
if __name__ == '__main__':
    data_path = '../input/'
    user_product_graph, prod_user_graph = read_graph_data(data_path + 'review_graph.txt')

    all_products = [k for k in prod_user_graph.keys()]
    with open(data_path + 'target_spammer.pickle', 'rb') as f:
        input_data = pickle.load(f)
    elite_users = input_data['elite_users']
    normal_users = input_data['normal_users']
    product_range = input_data['product_range']
    target_products = input_data['target_products']
    spammer_accounts = input_data['spammer_accounts']
    spammer_range = input_data['spammer_range']

    print("======Attack parameters======")
    print(f"number of possible targets = {len(product_range)}")
    print(f"number of targets = {len(target_products)}")

    print(f"number of possible spammers = {len(spammer_range)}")
    print(f"number of spammers = {len(spammer_accounts)}")
    print(f"\t elite = {sum([v for k, v in spammer_accounts.items()])}")
    print(f"\t normal = {len(spammer_accounts) - sum([v for k, v in spammer_accounts.items()])}")

    DeltaRevenues = []

    df = pd.read_csv(data_path + 'parameters.csv')

    for case in range(df.shape[0]):
        attack_size = df['num_spams_per_target'][case]
        num_spams = {}
        for target in target_products:
            num_spams[target] = attack_size

        attack_period = datetime.strptime(df['start_attack_date'][case], '%Y-%m-%d'),\
                        datetime.strptime(df['end_attack_date'][case], '%Y-%m-%d')
        attacking_spams, spam_graph = attack(user_product_graph, num_spams, spammer_accounts, attack_period)

        print(f"number of spams = {len(attacking_spams)}")
        top_k = df['top_k'][case]
        print(f'top k = {top_k}')
        evaluation(user_product_graph,
                   prod_user_graph,
                   elite_users,
                   product_range,
                   spammer_range,
                   target_products,
                   spammer_accounts,
                   attacking_spams,
                   detectPrior,
                   top_k)

        write_graph_data(spam_graph, f'submission_{case}.txt')