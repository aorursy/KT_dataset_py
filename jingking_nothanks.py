# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from random import sample

import random

from sklearn.tree import DecisionTreeRegressor as DTR

import pickle

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
class NT_Player:

    def __init__(self, name="random", Yes_Chance = 0.05):

        # print("Initiating Player", name)

        self.name = name

        self.strategy = (Yes_Chance, 1-Yes_Chance)



    def Take_a_Turn(self, player_status, current_card, current_coin, remaining_cards):

        # This function is required for any player class.

        # This is basically your "strategy". Decide whether to take the card based on the game status.

        moves = ["Yes, take the card.", "No, thanks."]

        return random.choices(moves, self.strategy)[0]

    

    def Get_Result(self, final_status):

        # This function is also required for any player class.

        self.result = final_status

        return None
class NT_Game:

    def __init__(self, players):

        self.player_count = len(players)

        self.players = players

        # print("Initiating a 5-player game.")

        self.deck = sample(list(range(3,36)), 24)

        player_initial_status = {"coins": 11, "cards": []}

        self.player_record = [{"coins": 11, "cards": []} for i in range(self.player_count)]        

        

    def Make_Player_Move(self, current_player):

        temp_list = self.player_record[current_player:] + self.player_record[:current_player]

        # Activate the Take_a_Turn function for a the current_player.

        # Provides it the current situation of the game and expects the player to return a Yes/No move.

        return self.players[current_player].Take_a_Turn(temp_list, self.current_card, 

                                                        self.current_coin, len(self.deck))

        

    def End_Game(self):

        # Eng the game. Calculate final scores, and activate the Get_Result function for all players.

        for p in self.player_record:

            score = p["coins"]

            for c in p["cards"]:

                if (c-1) not in p["cards"]:

                    score -= c

            p["final_score"] = score

        for current_player in range(self.player_count):

            temp_list = self.player_record[current_player:] + self.player_record[:current_player]

            self.players[current_player].Get_Result(temp_list)

            # Provide each player the final snapshot, and the final scores.

        scores = [p["final_score"] for p in self.player_record]

        winning_score = max(scores)

        self.results = [int(score==winning_score) for score in scores] 

        # Win/Loss outcome per player

        return None

    

    def Play_Game(self):

        # Start the game and run it to the end.

        current_player = 0

        self.current_coin = 0

        self.current_card = self.deck.pop(random.randint(0,len(self.deck)-1))

        error=False

        while len(self.deck)>0:

            if self.player_record[current_player]["coins"]==0:

                move = "Yes, take the card."

                # If you have no coins left, you must take the card.

                #print("Player", current_player, "is forced to take the card.")

            else:

                # Otherwise, you can move according to your strategy.

                move = self.Make_Player_Move(current_player)

            # Player makes a move.

            if move == "Yes, take the card.":

                self.player_record[current_player]["coins"] += self.current_coin

                self.current_coin = 0

                self.player_record[current_player]["cards"] += [self.current_card]

                self.current_card = self.deck.pop(random.randint(0,len(self.deck)-1))

            elif move == "No, thanks.":

                self.player_record[current_player]["coins"] -= 1

                self.current_coin += 1

            else:

                #print("Move Choice Error.")

                error = True

                break

            current_player = (current_player+1)%self.player_count

        if error:

            print("Game ended with an error.")

            return False

        else:

            #print("Game ended normally.")

            self.End_Game()

            return True
class Random_Recording_Player:

    def __init__(self, name="random", Yes_Chance = 0.05):

        # print("Initiating Player", name)

        self.name = name

        self.strategy = (Yes_Chance, 1-Yes_Chance)

        self.record = pd.DataFrame()



    def Add_Record(self, player_status, current_card, current_coin, remaining_cards, move):

        player_count = len(player_status)

        record = {"card": current_card, "coins": current_coin, "deck_size":remaining_cards}

        for i, player in zip(range(player_count), player_status):

            record["player_"+str(i)+"_coins"]= player["coins"]

            score = 0

            for c in player["cards"]:

                if (c-1) not in player["cards"]:

                    score -= c

            record["player_"+str(i)+"_card_score"]=score

            record["player_"+str(i)+"_plus"] = 0

            if   (current_card+1) in player["cards"]:

                record["player_"+str(i)+"_plus"] += 3

            elif (current_card+2) in player["cards"]:

                record["player_"+str(i)+"_plus"] += 1 

            record["player_"+str(i)+"_minus"] = 0

            if   (current_card-1) in player["cards"]:

                record["player_"+str(i)+"_minus"] += 3

            elif (current_card-2) in player["cards"]:

                record["player_"+str(i)+"_minus"] += 1

        if move=="Yes, take the card.":

            record["move"] = 1

        else:

            record["move"] = 0

        self.record = pd.concat( (self.record, pd.DataFrame([record])) )

        

    def Determine_Victory(self, final_status):

        own_score = final_status[0]["final_score"]

        for score in [p["final_score"] for p in final_status[1:]]:

            if score>own_score:

                return 0

        return 1

        

    def Take_a_Turn(self, player_status, current_card, current_coin, remaining_cards):

        moves = ["Yes, take the card.", "No, thanks."]

        move = random.choices(moves, self.strategy)[0]

        self.Add_Record(player_status, current_card, current_coin, remaining_cards, move)

        return move

    

    def Get_Result(self, final_status):

        self.result = final_status

        self.win = self.Determine_Victory(final_status)

        self.record["history"] = list(range(len(self.record)))[::-1]

        self.record["result"] = [None]*(len(self.record)-1) + [self.win]

        return None
class Central_Recorder:

    def __init__(self):

        self.record = pd.DataFrame()

        self.ID = 0

    

    def add_record(self, record):

        record["Record_ID"] = self.ID

        self.record = pd.concat( (self.record, record) )

        self.ID += 1
def Train_Model(data, old_data, features, target, max_depth, decay_rate=0.5):

    train_total = data.loc[data["history"]==0].set_index("Record_ID")

    for i in range(1000):

        model = DTR(max_depth = max_depth)

        model.fit(train_total[features], train_total[target])



        train = data.loc[data["history"]==i].set_index("Record_ID")

        Posi_Test = train.loc[:,:]

        Posi_Test.loc[:,"move"] = [1]*len(Posi_Test)

        train.loc[:,"Posi_Pred"] = model.predict(Posi_Test[features])

        Nega_Test = train.loc[:,:]

        Nega_Test.loc[:,"move"] = [0]*len(Nega_Test)

        train.loc[:,"Nega_Pred"] = model.predict(Nega_Test[features])

        temp = train.loc[:,[target] ]

        temp.loc[:, target] = train[["Posi_Pred", "Nega_Pred"]].max(axis=1)



        train1 = data.loc[data["history"]==i+1].set_index("Record_ID")

        if len(train1)==0:

            break

        train1.loc[:,target] = temp.loc[:,target] 

        train_total = pd.concat((train_total, train1))

    train_total = pd.concat((train_total, old_data.sample(frac=decay_rate)))

    model = DTR(max_depth = max_depth)

    model.fit(train_total[features], train_total[target])

    print( "Overall Accuracy", model.score(train_total[features], train_total[target]) )

    return model, train_total
class Tree_Player(NT_Player):

    def __init__(self, Model, name="Tree"):

        # print("Initiating Player", name)

        self.name = name

        self.Model = Model

        self.record = pd.DataFrame()



    def Make_Record(self, player_status, current_card, current_coin, 

                    remaining_cards, move, add=False):

        player_count = len(player_status)

        record = {"card": current_card, "coins": current_coin, "deck_size":remaining_cards}

        for i, player in zip(range(player_count), player_status):

            record["player_"+str(i)+"_coins"]= player["coins"]

            score = 0

            for c in player["cards"]:

                if (c-1) not in player["cards"]:

                    score -= c

            record["player_"+str(i)+"_card_score"]=score

            record["player_"+str(i)+"_plus"] = 0

            if   (current_card+1) in player["cards"]:

                record["player_"+str(i)+"_plus"] += 3

            elif (current_card+2) in player["cards"]:

                record["player_"+str(i)+"_plus"] += 1 

            record["player_"+str(i)+"_minus"] = 0

            if   (current_card-1) in player["cards"]:

                record["player_"+str(i)+"_minus"] += 3

            elif (current_card-2) in player["cards"]:

                record["player_"+str(i)+"_minus"] += 1

        if move=="Yes, take the card.":

            record["move"] = 1

        else:

            record["move"] = 0

        if add:

            self.record = pd.concat( (self.record, pd.DataFrame([record])) )

        return pd.DataFrame([record]) 

        

    def Take_a_Turn(self, player_status, current_card, current_coin, remaining_cards):

        Yes = self.Model.predict(self.Make_Record(player_status, current_card, current_coin, 

                                                  remaining_cards, "Yes, take the card."))

        No =  self.Model.predict(self.Make_Record(player_status, current_card, current_coin, 

                                                  remaining_cards, "No, thanks."))

        if Yes>No:

            move = "Yes, take the card."

        else:

            move = "No, thanks."

        self.Make_Record(player_status, current_card, current_coin, 

                         remaining_cards, "move", add=True)

        return move

    

    def Determine_Victory(self, final_status):

        own_score = final_status[0]["final_score"]

        for score in [p["final_score"] for p in final_status[1:]]:

            if score>own_score:

                return 0

        return 1

    

    def Get_Result(self, final_status):

        self.result = final_status

        self.win = self.Determine_Victory(final_status)

        self.record["history"] = list(range(len(self.record)))[::-1]

        self.record["result"] = [None]*(len(self.record)-1) + [self.win]

        return None
def Learn(models, model_names, NGames=10000, echo = 1000):

    Recorder = Central_Recorder()

    record = {"random_0.05": [0,0]}

    record.update({name: [0,0] for name in model_names})

    for i in range(NGames):

        players = [Random_Recording_Player("random_0.05", 0.05), 

                   Tree_Player(Model=models[0], name=model_names[0]),

                   Tree_Player(Model=models[0], name=model_names[0]),

                   Tree_Player(Model=models[1], name=model_names[1]),   

                   Tree_Player(Model=models[1], name=model_names[1])]

        players_setup = random.sample(players,5)

        g = NT_Game(players_setup)

        g.Play_Game()

        for p,r in zip(players_setup, g.results):

            record[p.name][1] += 1

            if r == 1:

                record[p.name][0] += 1

            Recorder.add_record(p.record)

        if echo and i%echo==0:

            print(i)

    print({k: record[k][0]/record[k][1] for k in record.keys()})

    return Recorder.record
filename = '../input/Model_Random.mdl'

Model = pickle.load(open(filename, 'rb'))
total_games = 20

echo = 0*total_games/10

max_depth = 20

decay_rate = 0.8

model_list = [Model, Model]

data = Learn(model_list, ["Tree_0", "Tree_0"], total_games, echo)

features = list(data.columns)[:-3]

target = "result"

old_data = pd.DataFrame()

for i in range(1,40):

    model, old_data = Train_Model(data, old_data, features, target, max_depth, decay_rate=decay_rate)

    model_list += [model]

    data = Learn(model_list[-2:], ["Tree_"+str(i-1), "Tree_"+str(i)], total_games, echo) 

    print(len(old_data))