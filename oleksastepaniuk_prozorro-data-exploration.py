import pandas as pd

import numpy as np

import os



import networkx as nx



from tqdm import tqdm_notebook

import tqdm



import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

%matplotlib inline



pd.options.display.float_format = '{:,.1f}'.format
# Load the initial data

data_dir = "/kaggle/input/prozorro-public-procurement-dataset/"

data_suppliers = "Suppliers.csv"

data_competitive = "Competitive_procurements.csv"



# # Check all data files

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



data = pd.read_csv(os.path.join(data_dir, data_competitive), index_col=0, dtype="str")

data[["lot_initial_value", "lot_final_value"]] = data[["lot_initial_value", "lot_final_value"]].astype(float)

data.index = pd.to_datetime(data.index)



# change variables format

data.loc[:, 'lot_announce_year'] = data.lot_announce_year.astype('int')

data.loc[:, 'supplier_dummy'] = data.supplier_dummy.astype('int')



print(f"The shape of the DF: {data.shape[0]:,.0f} rows, {data.shape[1]:,.0f} columns")

display(data.head(5).T)
years_list = data.lot_announce_year.unique()

procurement_types = data.lot_procur_type.unique()



print('Dataset includes:')

print(f'- {len(years_list)} years: {years_list}')

print(f'- First procurement was announced {data.index.min().date()} last procurement was announced {data.index.max().date()}\n')



print(f'- {data.lot_id.nunique():,.0f} competitive procurements [1]\n')



print(f'- {data.organizer_code.nunique():,.0f} entities that announced procurements - Organizers [2]\n')



print(f'- {data.participant_code.nunique():,.0f} companies that applied to participate in a competitive public procurement - Participants [3] \n')



print(f'- Initial value of a contract in the dataset varies from UAH {data.lot_initial_value.min():,.0f} to UAH {data.lot_initial_value.max():,.0f}.')

print(f'- Final value of a contract in the dataset varies from UAH {data.lot_final_value.min():,.0f} to UAH {data.lot_final_value.max():,.0f} [4]\n')



print(f'- {len(procurement_types)} procurement types: {", ".join(procurement_types)} [5]')

for procedure in procurement_types:

    procurements_number = len(data.query(f"lot_procur_type == '{procedure}'").lot_id.unique())

    print(f'   - {procedure}: {procurements_number:,.0f} procurements;')

print('\n')



print(f'- There are {data.lot_cpv_2_digs.nunique():,.0f} types of goods/services/works at the level of CPV 2 [6]')

print(f'- There are {data.lot_cpv_4_digs.nunique():,.0f} types of goods/services/works at the level of CPV 4')

print(f'- There are {data.lot_cpv.nunique():,.0f} types of goods/services/works at the level of CPV 8')
np.random.seed(12)

auction_list = data.lot_id.unique()

auction_number = np.random.randint(0, len(auction_list))

auction_target = auction_list[auction_number]







print(f'ID of the target auction is {auction_target}\n')



data_auction = data.query(f'lot_id == "{auction_target}"')



auction_date = data_auction.index[0].date()



organizer_name = data_auction.organizer_name[0]

organizer_code = data_auction.organizer_code[0]

organizer_region = data_auction.organizer_region[0]



object_cpv2 = data_auction.lot_cpv_2_digs[0]

object_cpv4 = data_auction.lot_cpv_4_digs[0]

object_cpv8 = data_auction.lot_cpv[0]



participants_number = len(data_auction)



winner_name = data_auction.query('supplier_dummy == 1').participant_name[0]

winner_code = data_auction.query('supplier_dummy == 1').participant_code[0]

winner_region = data_auction.query('supplier_dummy == 1').participant_region[0]



value_initial = data_auction.lot_initial_value[0]

value_final = data_auction.lot_final_value[0]







display(data_auction)



print(f'''Target auction was announced {auction_date} by {organizer_name} (Organizer ID {organizer_code}) from the {organizer_region} region. 

Organizer wanted to procure "{object_cpv2}". Object of procurement can be further clarified to "{object_cpv4}"  and "{object_cpv8}". 

Initial value of the contract determined by the Organizer was UAH {value_initial:,.0f}. In this case "{data_auction.lot_procur_type[0]}" procurement procedure was used.\n



{participants_number} companies apllied to compete for this contract. As a result of an auction, company {winner_name} (ID {winner_code}) from {winner_region} region offered UAH {value_final:,.0f} and became a winner.\n



The "nominal" savings estimate for the state budget can be calculated by comparing intial and final value of a contract. In this case UAH {np.round(value_initial - value_final, 2):,.0f} of budget money was saved as a result of competition between companies.''')
agregation_column = 'lot_cpv_2_digs'



# dataset were one row corresponds to one procurement

procurements_df = data.query('supplier_dummy == 1')



# calculate summary by CPV category

cpv_df = procurements_df.groupby(agregation_column).agg(Procurement_number=('lot_id', len), 

                                                        Allocated_budget_UAH=('lot_final_value', np.sum),

                                                        Contract_median_UAH=('lot_final_value', np.median))



# change allocated budget from UAH to million UAH

cpv_df.loc[:, 'Allocated_budget_mUAH'] = cpv_df.loc[:, 'Allocated_budget_UAH'] / 1000000

cpv_df.drop('Allocated_budget_UAH', axis=1, inplace=True)



# sort by allocated budget

cpv_df.sort_values('Allocated_budget_mUAH', ascending=False, inplace=True)



print('Top 5 CPV categories by the ammount of allocated budget:')

display(cpv_df.head(5))
####### Markets with max allocated budget, number of procurements, median contract



# max values of metrics

max_budget = cpv_df.Allocated_budget_mUAH.max()

max_procurements = cpv_df.Procurement_number.max()

max_median_contract = cpv_df.Contract_median_UAH.max()



# name of markets with max values

top_markets_names = {'budget': cpv_df.query(f'Allocated_budget_mUAH == {max_budget}').index[0],

                     'procurements': cpv_df.query(f'Procurement_number == {max_procurements}').index[0],

                     'contract': cpv_df.query(f'Contract_median_UAH == {max_median_contract}').index[0]}



# save info about each 'max' market

market_list = []

for market in top_markets_names.values():

    market_list.append((cpv_df.query(f'index == "{market}"').Allocated_budget_mUAH[0], 

                        cpv_df.query(f'index == "{market}"').Procurement_number[0],

                        market.split('_')[1]))





print(f'The most frequently procured object is "{top_markets_names["procurements"]}".')

print(f'The largest allocated budget (sum of all final values of contracts) belongs to "{top_markets_names["budget"]}".')

print(f'The largest median value of contract is associated with "{top_markets_names["contract"]}".')





####### Top X markets by allocated budget

markets_number = 3

top_budget = list(cpv_df.sort_values('Allocated_budget_mUAH', ascending=False).index[:markets_number])

    

# save info about each market

for market in top_budget:

    if market not in top_markets_names.values():

        market_list.append((cpv_df.query(f'index == "{market}"').Allocated_budget_mUAH[0], 

                            cpv_df.query(f'index == "{market}"').Procurement_number[0],

                            market.split('_')[1]))
fig, ax = plt.subplots(1,1,figsize=(9,5))

scatter = ax.scatter(x = cpv_df.Allocated_budget_mUAH,

                     y = cpv_df.Procurement_number, 

                     c = cpv_df.Contract_median_UAH/1000000,

                     alpha = 0.5)



ax.set_xlabel('Sum of contracts, m UAH')

ax.set_ylabel('Number of procurements')

# ax.set_title('Distribution of procurements by CPV 2 code')



ax.grid(True)

ax.legend(*scatter.legend_elements(), title='Median contract, m UAH', loc='lower right')



ax.set_yscale('log')

ax.set_xlim(-50000, cpv_df.Allocated_budget_mUAH.max()*1.25)





ax.axes.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

ax.axes.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))



# add key markets names

for market in market_list:

    ax.text(*market)



plt.show()
print(f"The total number of Organizers: {data['organizer_code'].nunique():,.0f}")

print(f"The total number of Participants: {data['participant_code'].nunique():,.0f}")



# Create the DF with distribution of organizers and participants by year

org_comp = pd.concat([data.groupby('lot_announce_year').organizer_code.nunique(), # the dynamic of the number of organizers

                      data.groupby('lot_announce_year').participant_code.nunique()], # the dynamic of the number of companies

                      axis=1)



# Plot the received distribution

participation_dynamics = org_comp.plot.bar()

participation_dynamics.axes.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))



plt.ylabel('Number of entities')

plt.xlabel(None)



plt.show()
plt.figure(figsize=(7,5))



#### Prepare data 



# number of procurements per year

procurement_per_year = data.groupby("lot_announce_year").agg(Procurement_number = ("lot_id", 'nunique'))



# number of unique Participants per each auction

df_ave_num_comp = data.groupby(["lot_id","lot_announce_year"]).participant_code.nunique().reset_index()



# average number of participants per auction for each year

df_ave_num_comp = df_ave_num_comp.groupby("lot_announce_year").participant_code.mean()





#### Create plot



plot_lots = procurement_per_year.Procurement_number.plot.line(label='Procurements number', marker='o')

plot_lots.locator_params(integer=True)

plot_lots.axes.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

plt.ylabel('Number of procurements')



plot_participants_average = df_ave_num_comp.plot.line(secondary_y=True, label='Average number of participants (right axis)', marker='o')

plt.ylabel('Average number of participants per procurement')



# combine legends

plot_participants_average.xaxis.label.set_visible(False)

plot_lots.xaxis.label.set_visible(False)



lines, labels = plot_lots.get_legend_handles_labels()

lines2, labels2 = plot_participants_average.get_legend_handles_labels()

plot_participants_average.legend(lines + lines2, labels + labels2, loc=2)



plt.show()
target_cpv_code = "03110000-5_Crops, products of market gardening and horticulture"

target_year = 2015



df = data.query(f'lot_cpv_4_digs == "{target_cpv_code}" & lot_announce_year == "{target_year}"')

df = df.sort_values('participant_code')

df.loc[:, 'contracts_value'] = np.where(df.supplier_dummy == 1, df.lot_final_value, 0)
def create_summary(df):

    '''

    Returns pivot table were each row corresponds to one Participant.

    For each Participant table provides participant_name, total value of won contracts (contracts_value) and 

    number of won contracts (contracts_number).

    In addition, function creates variables necessary for the graph visualization - nodes label, size and color.

    '''

    

    #### Create summary

    

    df_summary = df.groupby('participant_code').agg(participant_name = ('participant_name', 'first'),

                                                contracts_value = ('contracts_value', 'sum'),

                                                contracts_number = ('supplier_dummy', 'sum'))





    #### Variables necessary for the visialization



    # add IDs of the top 5 companies by the total value of contracts to the graph

    top_5_companies = df_summary.sort_values('contracts_value', ascending=False)['participant_name'].values[:5]

    df_summary.loc[:, 'label'] = np.where(df_summary.participant_name.isin(top_5_companies), df_summary.index, '')



    # assign size categories

    # if company won less than 5% of total value of contracts, its category is 100

    # if its share is between 5% and 10% - 200; between 10% and 15% - 300, etc.

    df_summary.loc[:, 'contracts_share'] = df_summary.contracts_value / df_summary.contracts_value.sum() * 100

    df_summary.loc[:, 'size_category'] = (df_summary.contracts_share // 5 + 1) * 100

    df_summary.loc[:, 'size_category'] = np.where(df_summary.contracts_number==0, 30, df_summary.size_category)



    # assign red color is company did not won any contracts, else assign blue

    df_summary.loc[:, 'color'] = np.where(df_summary.contracts_number == 0, 'r', 'b')

    

    return df_summary
df_summary = create_summary(df)

top_10_companies_2015 = df_summary.sort_values('contracts_value', ascending=False)[:10]



contracts_max_number = df_summary['contracts_number'].max()

contracts_max_number_name = df_summary.query(f'contracts_number == "{contracts_max_number}"')['participant_name'].values



contracts_max_value = df_summary['contracts_value'].max()

contracts_max_value_name = df_summary.query(f'contracts_value == "{contracts_max_value}"')['participant_name'].values





print('Summary table for the Partiticpants (head):')

display(df_summary.head(8))



print(f'''In {target_year} {len(df_summary)} companies competed for "{target_cpv_code}" contracts.

During the year there were {df.lot_id.nunique()} auctions with total value of contracts equal to UAH {df_summary.contracts_value.sum():,.0f}.

The company that won the largest number of auctions was "{'", "'.join(contracts_max_number_name)}" ({contracts_max_number} contracts).

The company with the largest total value of contracts is "{'", "'.join(contracts_max_value_name)}" (UAH {contracts_max_value:,.0f}).

{sum(df_summary.contracts_number==0)} companies participated but did not won any contracts.''')
# Function  for creating the network

def making_graph_1_mode(df):

    """The function takes the df and creates the 1-mode network of tender participants.

       Node - tender participant, edge - participation in tender organized by particular public entity.

       For example, the two companies are connected if they both particpated in tender organized by one public entity"""

    

    # Create the table where columns are public entities codes (organizer_code) and index - tender participants codes (participant_code)

    df = df.pivot_table(values = "lot_final_value", index="participant_code", columns="organizer_code", aggfunc="count").fillna(0)

    

    # Dot product that 'connects' all the participants

    df = df.dot(df.T)

    

    # Simplification of the received matrix

    df = df.astype(int)

    np.fill_diagonal(df.values, 0)

    

    # Create the graph from the received adjacency matrix

    G = nx.from_pandas_adjacency(df)

    

    return G
G = making_graph_1_mode(df)



# Let's make some calculation based on the graph created:

num_nodes = G.number_of_nodes()

print("The number of nodes (companies)) is: ", num_nodes)



num_edges = G.number_of_edges()

print("The number of edges (participations in the same auction) is: ", num_edges)
plt.figure(figsize=(12,6)) 

nx.draw(G,

        labels = df_summary.label,

        node_size = df_summary.size_category,

        node_color = df_summary.color,

        pos = nx.nx_pydot.graphviz_layout(G),

        font_size = 14, 

        alpha = 0.8)



plt.title(f'"{target_cpv_code}" auctions in {target_year}', fontsize=18)

plt.show()
# Select the data we need from the initial dataset

target_year = 2019



df = data.query(f'lot_cpv_4_digs == "{target_cpv_code}" & lot_announce_year == "{target_year}"')

df = df.sort_values('participant_code')

df.loc[:, 'contracts_value'] = np.where(df.supplier_dummy == 1, df.lot_final_value, 0)



# summary tables

df_summary = create_summary(df)

top_10_companies_2019 = df_summary.sort_values('contracts_value', ascending=False)[:10]



# Create the graph from the data selected

G = making_graph_1_mode(df)



num_nodes = G.number_of_nodes()

print("The number of nodes (companies)) is: ", num_nodes)



num_edges = G.number_of_edges()

print("The number of edges (participations in the same auction) is: ", num_edges)
plt.figure(figsize=(12,6)) 

nx.draw(G,

        labels = df_summary.label,

        node_size = df_summary.size_category,

        node_color = df_summary.color,

        pos = nx.nx_pydot.graphviz_layout(G),

        font_size = 14, 

        alpha = 0.8)



plt.title(f'"{target_cpv_code}" auctions in {target_year}', fontsize=18)

plt.show()
top_10_companies_2015 = top_10_companies_2015[['participant_name', 'contracts_value', 'contracts_number', 'contracts_share']]

top_10_companies_2015.columns = ['participant_name_2015', 'contracts_value_2015', 'contracts_number_2015', 'contracts_share_2015']



top_10_companies_2019 = top_10_companies_2019[['participant_name', 'contracts_value', 'contracts_number', 'contracts_share']]

top_10_companies_2019.columns = ['participant_name_2019', 'contracts_value_2019', 'contracts_number_2019', 'contracts_share_2019']





top_10_companies = pd.merge(top_10_companies_2015, top_10_companies_2019, right_index=True, left_index=True, how='outer')



top_10_companies.loc[:, 'participant_name'] = np.where(top_10_companies.participant_name_2015.isna(),

                                                       top_10_companies.participant_name_2019,

                                                       top_10_companies.participant_name_2015)



top_10_companies = top_10_companies[['participant_name',

                                     'contracts_value_2015', 'contracts_value_2019',

                                     'contracts_number_2015', 'contracts_number_2019',

                                     'contracts_share_2015', 'contracts_share_2019']]



top_10_companies_micolumns = pd.MultiIndex.from_tuples([('participant_name', ''), 

                                                        ('contracts_value', '2015'), ('contracts_value', '2019'),

                                                        ('contracts_number', '2015'), ('contracts_number', '2019'),

                                                        ('contracts_share', '2015'), ('contracts_share', '2019'),],

                                                        names=['variable', 'year'])



top_10_companies.columns = top_10_companies_micolumns

top_10_companies.fillna('', inplace=True)



display(top_10_companies)
def making_figure_1_mode(cpv_code):

    years_list = data["lot_announce_year"].unique().tolist()

    

    fig, axes = plt.subplots(nrows=6, ncols=1, figsize = (15, 25))

    

    # Set super title

    fig.suptitle(f"CPV code: {cpv_code}", size=18)



    for i, ax in enumerate(axes.flatten()):

        

        if i >= len(years_list):

            ax.axis('off')

        else:

            year = years_list[i]

            

            # Set title

            ax.set_title(year, size=14, loc="left")

            

            # Select appropriate dDF

            df = data.query(f'lot_cpv_4_digs == "{cpv_code}" and lot_announce_year == "{year}"')

            df = df.sort_values('participant_code')

            df.loc[:, 'contracts_value'] = np.where(df.supplier_dummy == 1, df.lot_final_value, 0)

            

            # Get summary per Participant

            df_summary = df_summary = create_summary(df)

            

            # Create Graph from DF

            G = making_graph_1_mode(df)

            

            # Plot graph

            nx.draw(G,

                    labels = df_summary.label,

                    node_size = df_summary.size_category,

                    node_color = df_summary.color,

                    pos=nx.nx_pydot.graphviz_layout(G),

                    font_size = 14, 

                    alpha = 0.3,

                    font_weight="bold",

                    ax=ax)
# to see details open image in a separate tab

making_figure_1_mode(target_cpv_code)