import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap



%matplotlib inline
known_behaviors = pd.read_csv("https://raw.githubusercontent.com/vberaudi/utwt/master/known_behaviors2.csv")

known_behaviors.head()
a = known_behaviors[known_behaviors.Mortgage == 1]

b = known_behaviors[known_behaviors.Pension == 1]

c = known_behaviors[known_behaviors.Savings == 1]

print("Number of clients: %d" %len(known_behaviors))

print("Number of clients predicted to buy mortgage accounts: %d" %len(a))

print("Number of clients predicted to buy pension accounts: %d" %len(b))

print("Number of clients predicted to buy savings accounts: %d" %len(c))
known_behaviors["nb_products"] = known_behaviors.Mortgage + known_behaviors.Pension + known_behaviors.Savings
abc = known_behaviors[known_behaviors.nb_products > 1]

print("We have %d clients who bought several products" %len(abc))

abc = known_behaviors[known_behaviors.nb_products == 3]

print("We have %d clients who bought all the products" %len(abc))
products = ["Savings", "Mortgage", "Pension"]
def plot_cloud_points(df):

    figure = plt.figure(figsize=(20, 5))

    my_cm  = ListedColormap(['#bb0000', '#00FF00'])

    axes = {p : ('age', 'income') if p != "Mortgage"else ('members_in_household', 'loan_accounts') for p in products}

    for product in products:

        ax = plt.subplot(1, len(products), products.index(product)+1)

        ax.set_title(product)

        axe = axes[product]

        plt.xlabel(axe[0])

        plt.ylabel(axe[1])

        ax.scatter(df[axe[0]], df[axe[1]], c=df[product], cmap=my_cm, alpha=0.5)
plot_cloud_points(known_behaviors)
known_behaviors.columns
cols = ['age', 'income', 'members_in_household', 'loan_accounts']
X = known_behaviors[cols]

ys = [known_behaviors[p] for p in products]
X.head()
from sklearn import svm

from sklearn import ensemble
classifiers = []

for i,p in enumerate(products):

    clf = ensemble.GradientBoostingClassifier()

    clf.fit(X, ys[i])

    classifiers.append(clf)
unknown_behaviors = pd.read_csv("https://raw.githubusercontent.com/vberaudi/utwt/master/unknown_behaviors.csv")
for c in unknown_behaviors.columns:

    assert c in known_behaviors.columns
to_predict = unknown_behaviors[cols]
print("Number of new customers: %d" %len(unknown_behaviors))
import warnings

warnings.filterwarnings('ignore')
predicted = [classifiers[i].predict(to_predict) for i in range(len(products))]

for i,p in enumerate(products):

    to_predict[p] = predicted[i]

to_predict["id"] = unknown_behaviors["customer_id"]
offers = to_predict

offers.head()
offers = offers.rename_axis('index_nb').reset_index()
a = offers[offers.Mortgage == 1]

b = offers[offers.Pension == 1]

c = offers[offers.Savings == 1]

print("Number of new customers: %d" %len(offers))

print("Number of customers predicted to buy mortgages: %d" %len(a))

print("Number of customers predicted to buy pensions: %d" %len(b))

print("Number of customers predicted to buy savings: %d" %len(c))
to_predict["nb_products"] = to_predict.Mortgage + to_predict.Pension + to_predict.Savings



abc = to_predict[to_predict.nb_products > 1]

print("We predicted that %d clients would buy more than one product" %len(abc))

abc = to_predict[to_predict.nb_products == 3]

print("We predicted that %d clients would buy all three products" %len(abc))
# How much revenue is earned when selling each product

productValue = [200, 300, 400]

value_per_product = {products[i] : productValue[i] for i in range(len(products))}



# Total available budget

availableBudget = 25000



# For each channel, cost of making a marketing action and success factor

channels =  pd.DataFrame(data=[("gift", 20.0, 0.20), 

                               ("newsletter", 15.0, 0.05), 

                               ("seminar", 23.0, 0.30)], columns=["name", "cost", "factor"])



offersR = range(0, len(offers))

productsR = range(0, len(products))

channelsR = range(0, len(channels))
gsol = pd.DataFrame()

gsol['id'] = offers['id']



budget = 0

revenue = 0



for product in products:

    gsol[product] = 0



noffers = len(offers)



# ensure the 10% per channel by choosing the most promising per channel

for c in channelsR: #, channel in channels.iterrows():

    i = 0;

    while (i< ( noffers // 10 ) ):

        # find a possible offer in this channel for a customer not yet done

        added = False

        for o  in offersR:

            already = False

            for product in products:   

                if gsol.get_value(index=o, col=product) == 1:

                    already = True

                    break

            if already:

                continue

            possible = False

            possibleProduct = None

            for product in products:

                if offers.get_value(index=o, col=product) == 1:

                    possible = True

                    possibleProduct = product

                    break

            if not possible:

                continue

            #print "Assigning customer ", offers.get_value(index=o, col="id"), " with product ", product, " and channel ", channel['name']

            gsol.set_value(index=o, col=possibleProduct, value=1)

            i = i+1

            added = True

            budget = budget + channels.get_value(index=c, col="cost")

            revenue = revenue + channels.get_value(index=c, col="factor")*value_per_product[product]            

            break

        if not added:

            print("NOT FEASIBLE")

            break
# add more to complete budget       

while (True):

    added = False

    for c, channel in channels.iterrows():

        if (budget + channel.cost > availableBudget):

            continue

        # find a possible offer in this channel for a customer not yet done

        for o  in offersR:

            already = False

            for product in products:   

                if gsol.get_value(index=o, col=product) == 1:

                    already = True

                    break

            if already:

                continue

            possible = False

            possibleProduct = None

            for product in products:

                if offers.get_value(index=o, col=product) == 1:

                    possible = True

                    possibleProduct = product

                    break

            if not possible:

                continue

            #print "Assigning customer ", offers.get_value(index=o, col="id"), " with product ", product, " and channel ", channel['name']

            gsol.set_value(index=o, col=possibleProduct, value=1)

            i = i+1

            added = True

            budget = budget + channel.cost

            revenue = revenue + channel.factor*value_per_product[product]            

            break

    if not added:

        print("FINISH BUDGET")

        break

    

print(gsol.head())
a = gsol[gsol.Mortgage == 1]

b = gsol[gsol.Pension == 1]

c = gsol[gsol.Savings == 1]



abc = gsol[(gsol.Mortgage == 1) | (gsol.Pension == 1) | (gsol.Savings == 1)]



print("Number of clients: %d" %len(abc))

print("Numbers of Mortgage offers: %d" %len(a))

print("Numbers of Pension offers: %d" %len(b))

print("Numbers of Savings offers: %d" %len(c))

print("Total Budget Spent: %d" %budget)

print("Total revenue: %d" %revenue)





comp1_df = pd.DataFrame(data=[["Greedy", revenue, len(abc), len(a), len(b), len(c), budget]], columns=["Algorithm","Revenue","Number of clients","Mortgage offers","Pension offers","Savings offers","Budget Spent"])
from __future__ import print_function

from ortools.linear_solver import pywraplp



solver = pywraplp.Solver('SolveMCProblemMIP',

                           pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
channelVars = {}



# variables

for o in offersR:

    for p in productsR:

        for c in channelsR:

            channelVars[o,p,c] = solver.BoolVar('channelVars[%i,%i,%i]' % (o,p,c))
# constraints

# At most 1 product is offered to each customer

for o in offersR:

    solver.Add(solver.Sum(channelVars[o,p,c] for p in productsR for c in channelsR) <=1)



# Do not exceed the budget

solver.Add(solver.Sum(channelVars[o,p,c]*channels.get_value(index=c, col="cost") 

                                           for o in offersR 

                                           for p in productsR 

                                           for c in channelsR)  <= availableBudget)



# At least 10% offers per channel

for c in channelsR:

    solver.Add(solver.Sum(channelVars[o,p,c] for p in productsR for o in offersR) >= len(offers) // 10)
print(f'Number of constraints : {solver.NumConstraints()}' )

print(f'Number of variables   : {solver.NumVariables()}')
# objective 

obj = 0



for c in channelsR:

    for p in productsR:

        product=products[p]

        coef = channels.get_value(index=c, col="factor") * value_per_product[product]

        obj += solver.Sum(channelVars[o,p,c] * coef * offers.get_value(index=o, col=product) for o in offersR)



solver.Maximize(obj)
# time limit

#solver.set_time_limit = 100.0
sol = solver.Solve()
totaloffers = solver.Sum(channelVars[o,p,c] for o in offersR for p in productsR for c in channelsR)



budgetSpent = solver.Sum(channelVars[o,p,c]*channels.get_value(index=c, col="cost") 

                                           for o in offersR 

                                           for p in productsR 

                                           for c in channelsR)



print(f'Total offers : {totaloffers.solution_value()}')

print(f'Budget Spent : {budgetSpent.solution_value()}')



for c, n in zip(channelsR, list(channels.name)):

    channel_kpi = solver.Sum(channelVars[o,p,c] for p in productsR for o in offersR)

    print(f'{n} : {channel_kpi.solution_value()}')



for p, n in zip(productsR, products):

    product = products[p]

    product_kpi = solver.Sum(channelVars[o,p,c] for c in channelsR for o in offersR)

    print(f'{n} : {product_kpi.solution_value()}')



print(f'It has taken {solver.WallTime()} milliseconds to solve the optimization problem.')
from itertools import product as prod



results = []



for o, p, c in prod(list(range(len(offers))),list(range(len(products))), list(range(len(channels)))):

    if channelVars[(o, p, c)].solution_value() > 0:

        #print(f'{o} : {products[p]}')

        results.append([o, products[p]])



results = pd.DataFrame(results, columns=['index_nb', 'product'])
results.head()
results['product'].value_counts()
all_results = offers.merge(results, on='index_nb', how='inner')

all_results.head()