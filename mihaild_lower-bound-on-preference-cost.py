import numpy as np

import pandas as pd



from ortools.graph.pywrapgraph import SimpleMinCostFlow
DAYS = 100

FAMILIES = 5000



MIN_PER_DAY = 125

MAX_PER_DAY = 300



# ortools doesn't support float penalties, so we will multiply all family penalties by lcm of possible families sizes before dividing by family size

PENALTY_MULTIPLIER = 840
data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')
# from https://www.kaggle.com/nickel/santa-s-2019-fast-pythonic-cost-30-s

penalties = np.asarray([

    [

        0,

        50,

        50 + 9 * n,

        100 + 9 * n,

        200 + 9 * n,

        200 + 18 * n,

        300 + 18 * n,

        300 + 36 * n,

        400 + 36 * n,

        500 + 36 * n + 199 * n,

        500 + 36 * n + 398 * n

    ] for n in range(data.n_people.max() + 1)

])
per_person_penalties = (penalties * PENALTY_MULTIPLIER // np.tile(np.arange(1, penalties.shape[0] + 1), [11, 1]).T)

# ortools needs int, not numpy.int64

per_person_penalties = per_person_penalties.tolist()
sizes = data.n_people.values.tolist()

preferences = data.drop('n_people', axis=1).values - 1 # days start from 0
family_nodes = list(range(FAMILIES))

day_nodes = list(range(FAMILIES, FAMILIES + DAYS))

mandatory_sink_node = FAMILIES + DAYS

extra_sink_node = mandatory_sink_node + 1
G = SimpleMinCostFlow()



for f in range(FAMILIES):

    for pref_num, day in enumerate(preferences[f]):

        G.AddArcWithCapacityAndUnitCost(family_nodes[f], day_nodes[day], 10, per_person_penalties[sizes[f]][pref_num])

    G.AddArcWithCapacityAndUnitCost(family_nodes[f], mandatory_sink_node, 10, per_person_penalties[sizes[f]][-1])

    G.AddArcWithCapacityAndUnitCost(family_nodes[f], extra_sink_node, 10, per_person_penalties[sizes[f]][-1])



for d in range(DAYS):

    G.AddArcWithCapacityAndUnitCost(day_nodes[d], mandatory_sink_node, MIN_PER_DAY, 0)

    G.AddArcWithCapacityAndUnitCost(day_nodes[d], extra_sink_node, MAX_PER_DAY - MIN_PER_DAY, 0)



for f in range(FAMILIES):

    G.SetNodeSupply(family_nodes[f], sizes[f])



min_supply = MIN_PER_DAY * DAYS

G.SetNodeSupply(mandatory_sink_node, -min_supply)

G.SetNodeSupply(extra_sink_node, -(sum(sizes) - min_supply))



assert G.Solve() == G.OPTIMAL
print("Lower bound on cost: ", G.OptimalCost() / PENALTY_MULTIPLIER)
import seaborn as sns

import matplotlib.pyplot as plt



pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 150)
days_choices = np.zeros((DAYS, preferences.shape[1]), 'int')

for arc_num in range(G.NumArcs()):

    if G.Tail(arc_num) < FAMILIES and G.Head(arc_num) < FAMILIES + DAYS: # arc from family node to day node

        f = G.Tail(arc_num)

        d = G.Head(arc_num) - FAMILIES

        choice = np.where(preferences[f] == d)[0][0]

        days_choices[d][choice] += G.Flow(arc_num)
print(sum(sizes) - days_choices.sum())
print(days_choices.sum(axis=0), days_choices[:, 1:].sum())
agg_data = pd.DataFrame(np.cumsum(days_choices, axis=1), columns=['n_people_' + str(i) for i in range(10)])

agg_data['assigned_day'] = np.arange(DAYS)
f, ax = plt.subplots(figsize=(12, 20))

sns.set_color_codes("pastel")

sns.barplot(x='n_people_4', y='assigned_day', data=agg_data, label='choice_4', orient='h', color='r')

sns.barplot(x='n_people_3', y='assigned_day', data=agg_data, label='choice_3', orient='h', color='g')

sns.barplot(x='n_people_2', y='assigned_day', data=agg_data, label='choice_2', orient='h', color='y')

sns.barplot(x='n_people_1', y='assigned_day', data=agg_data, label='choice_1', orient='h', color='c')

sns.barplot(x='n_people_0', y='assigned_day', data=agg_data, label='choice_0', orient='h', color='b')

ax.axvline(125, color="k", clip_on=False)

ax.axvline(300, color="k", clip_on=False)

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlabel="Occupancy")

plt.show()