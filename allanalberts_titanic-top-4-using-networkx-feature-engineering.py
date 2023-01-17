import pandas as pd
import numpy as np
import re 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import networkx as nx
from itertools import combinations

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Visualize the Decision Tree Classifier
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv("../input/train.csv")
test =  pd.read_csv('../input/test.csv')

# join train and test datasets for feature engineering:
data = pd.concat([train, test])

# save the length of original training dataset to use later for splitting back into training and test datasets
train_len = len(train)

# Check for missing values
print('Missing Values:')
data = data.fillna(np.nan)
data.isnull().sum()
data['Title'] = data.Name.str.extract(' ([A-Z][a-z]+)\.', expand=False)
sns.swarmplot(y='Age', x='Title', hue='Survived', data=data[(data.Sex=='male')&(data.Age<17)]);
plt.title('Survival of Young Men');
plt.xlabel('Title (from Name field)');
plt.legend(['Survived', 'Died'],loc='lower right');
sns.despine()
data['Ptype_AdultM'] = np.where((data.Sex=='male') & (data.Title!='Master'), 1, 0)
data['Ptype_F_Ch'] = np.where(data.Ptype_AdultM==0, 1, 0)
data['LastName'] = data.Name.str.extract('([A-Za-z]+),', expand=False)
data['HyphenName'] = data.Name.str.extract('([A-Za-z]+)-', expand=False)
data['MaidenName1'] = np.where(data.Title=='Mrs', data.Name.str.extract('([A-Za-z]+)\)', expand=False), np.NaN)
data['MaidenName2'] = np.where(data.Title=='Mrs', data.Name.str.extract('([A-Za-z]+)$', expand=False), np.NaN)
# process tickets that end in a numeric sequence:
data['TicketNum'] = data.Ticket.str.extract('\s([0-9]+)', expand=False)

# process tickets that do not end in a numeric sequence:
data['TicketNum'] = np.where(pd.isnull(data.TicketNum), data.Ticket.str.extract('([0-9]+)', expand=False), data.TicketNum)

# assign zero where the ticket does not contain numbers:
data['TicketNum'] = data.TicketNum.fillna(0).astype(int)
def family_names(Passenger):
    '''Returns a list of potential family names for a given passenger'''
    NameList = []
    NameList.append(G.node[Passenger]['LastName'])
    if pd.notnull(G.node[Passenger]['HyphenName']):
        NameList.append(G.node[Passenger]['HyphenName'])
    if pd.notnull(G.node[Passenger]['MaidenName1']):
        NameList.append(G.node[Passenger]['MaidenName1'])
    elif pd.notnull(G.node[Passenger]['MaidenName2']):
            NameList.append(G.node[Passenger]['MaidenName2'])
    return NameList
def name_match(Passenger1, Passenger2):
    '''Compares two passengers and returns TRUE if they share
       the same class and family name'''
    NameList_p1 = family_names(Passenger1)
    NameList_p2 = family_names(Passenger2)
    Pclass_p1 = G.node[Passenger1]['Pclass']
    Pclass_p2 = G.node[Passenger2]['Pclass']
    if Pclass_p1==Pclass_p2:
        return bool(set(NameList_p1).intersection(set(NameList_p2)))
    else:
        return False
def similar_ticket(Passenger1, Passenger2, TicketProximity):
    '''Compares two passenger tickets and returns TRUE if they
       are sequentially within the value defined by TicketProximity'''
    TicketNum_p1 = G.node[Passenger1]['TicketNum']
    TicketNum_p2 = G.node[Passenger2]['TicketNum']
    SequentialLimit = abs(TicketNum_p1-TicketNum_p2)
    return bool((SequentialLimit!=0)&(SequentialLimit<=TicketProximity))
def find_match(PassengerGroup, TargetGroup, TicketProximity, NameCheck):
    '''Finds passengers in PassengerGroup list that have a matching family member in the TargetGroup list.
       NameCheck=True indicates a family name comparison should be used and TicketProximity indicates how close the ticket
       number sequence must be for a passenger match. TicketProximity=0 indicates no ticket comparison should be used.
       Returns a list of passengers that still need to be matched to additional passengers.'''
    NodeList = []
    Append = False
    for Passenger in PassengerGroup:   
        try:  #remove passenger if in target group
            TargetGroup.remove(Passenger)
        except ValueError:
            pass  # do nothing!

        for Passenger2 in TargetGroup:
            if NameCheck:
                if name_match(Passenger,Passenger2):
                    Append = True
                    if TicketProximity > 0:
                        if not similar_ticket(Passenger, Passenger2, TicketProximity):
                            Append = False
            else:
                if TicketProximity > 0:
                    if similar_ticket(Passenger, Passenger2, TicketProximity):
                        Append = True

            if Append:
                NodeList.append(Passenger)
                NodeList.append(Passenger2)
                Append = False
        
    # make a list of tuples by creating an interator and zipping it with itself
    it = iter(NodeList) 
    EdgeList = list(zip(it, it))
    
    if pd.notnull(EdgeList).any():  # add edges to graph for a passenger match
        for u, v in EdgeList:
            G.add_edge(u,v)
    
    # Print updated Networked stats
    PassengersInNetworks = [n for n in G if G.degree(n)>0]
    print('Passengers in Networks: {}'.format(len(PassengersInNetworks)))
    print('Percentage in Networks: {:.0%}'.format(len(PassengersInNetworks)/len(G.nodes)))

    # Find passengers whose family size excedes the size of their network (still need to find edges to family members)
    PassengerNetTooSmall = [n for n, d in G.nodes(data=True) if (G.degree(n)<(d['Parch']+d['SibSp']))]
    print('\nPassengers in a network that is still smaller than their family size: {}'.format(len(PassengerNetTooSmall)))
    return PassengerNetTooSmall
G = nx.Graph()
all_nodes = data.PassengerId.values

# set the index equal to the node key(PassengerId)
data.set_index('PassengerId', inplace=True)

# define individual node attributes and add node to graph
for n in all_nodes:
    G.add_node(n, \
               TicketNum=data.loc[n].TicketNum, \
               Pclass=data.loc[n].Pclass, \
               Parch=data.loc[n].Parch, \
               SibSp=data.loc[n].SibSp, \
               LastName=data.loc[n].LastName, \
               MaidenName1=data.loc[n].MaidenName1, \
               MaidenName2=data.loc[n].MaidenName2, \
               HyphenName=data.loc[n].HyphenName)
data.reset_index(inplace=True) 
# Step1: Assignment based solely on shared ticket numbers

TicketList = list(data.Ticket.unique())
for T in TicketList:
    data2 = list(data[data.Ticket==T].PassengerId)
    for u, v in combinations(data2, 2):
        G.add_edge(u,v)    
        
print('Total Passengers: {}'.format(len(G.nodes)))
PassengersInNetworks = [n for n in G if G.degree(n)>0]
print('Passengers in Networks: {}'.format(len(PassengersInNetworks)))
print('Percentage in Networks: {:.0%}'.format(len(PassengersInNetworks)/len(G.nodes)))

# Find passengers whose family size excedes the size of their network (still need to find edges to family members)
PassengerNetTooSmall = [n for n, d in G.nodes(data=True) if (G.degree(n)<(d['Parch']+d['SibSp']))]
print('\nPassengers in a network that is still smaller than their family size: {}'.format(len(PassengerNetTooSmall)))
# Step2: check for matching family name AND ticket sequentially within 10 tickets of each other 

PassengerGroup = list(set(G.nodes()).copy())
TargetGroup = PassengerGroup.copy()
TicketProximity = 10
PassengerNetTooSmall = find_match(PassengerGroup, TargetGroup, TicketProximity, NameCheck=True)
# Step3: check for matching family name (irregardless of ticket sequence) for Passengers needing Family Network assginments.

PassengerGroup = PassengerNetTooSmall.copy()
TargetGroup = list(set(G.nodes()).copy() )
TicketProximity = 0
PassengerNetTooSmall = find_match(PassengerGroup, TargetGroup, TicketProximity, NameCheck=True)
# Step4: check for sequentially issued tickets that each have corresponding missing family members (name mispelling issues)
PassengerGroup = PassengerNetTooSmall.copy()
TargetGroup = PassengerGroup.copy()
TicketProximity = 1
PassengerNetTooSmall = find_match(PassengerGroup, TargetGroup, TicketProximity, NameCheck=False)
print(PassengerNetTooSmall)
print('Married Couple with missing values in SibSp field:')
data[data.Ticket=='364498'][['Ticket','Name','Age','Parch','SibSp']]
# Remove unconnected passenger nodes from graph

nodelist = []
for n in G.nodes:
    if (G.degree(n)>0):
        nodelist.append(n)
G.remove_nodes_from([n for n in G if n not in nodelist])
# create a dictionary of networks with the key being the new Network number and the value being a list with the passengers
# as the first element and the size of the Network as the second element.
network_members = dict()
n = 1
for g in list(nx.connected_component_subgraphs(G)):
    network_members[n] = [g.nodes(), len(g.nodes())]
    n += 1

# initialize all passengers to default for passengers not in traveling in groups (Network=0)
data['NetworkNum'] = 0 
data['NetSize'] = 1

# loop through the dictionary and assign the Network number and size to each Passenger:
data.set_index('PassengerId', inplace=True)
for key, value in network_members.items():
    for p in value[0]:
        data.loc[p, 'NetworkNum'] = key
        data.loc[p, 'NetSize'] = value[1]
data.reset_index(inplace=True) 

# Create feature InNetwork=1 if passenger is traveling with other people.
data['InNetwork'] = np.where(data.NetworkNum==0, 0, 1)
# Create supporting passenger feature, Status, with values: -1=status dead, 0=unknown status, 1=status survived
data['Status'] = data.Survived.fillna(0)  # unknown survival status=0
data['Status'] = np.where(data.Survived==0, -1, data.Status)
# For each passenger that is in a Network (NetworkNum!=0), update Network average status for the passengers
# that are of this type, exclusive of the current passenger.

def update_passenger_network_status(data, Type):
    NstatusField = 'Nstatus' + Type
    PtypeField = 'Ptype' + Type
    
    data[NstatusField] = 0
    data.set_index('PassengerId', inplace=True)
    
    # step through each passenger network
    for n in data[data.NetworkNum!=0].NetworkNum.unique():

        # calculate the Network totals for this passenger type inclusive of the all qaulified passengers
        NetworkSum = data[(data.NetworkNum==n)&(data[PtypeField]==1)].Status.sum()
        NetworkSize = data[data.NetworkNum==n][PtypeField].sum()

        # step though each passenger in the network and update their network status
        for index, row in data[(data.NetworkNum==n)].iterrows():
            PNetworkSum = NetworkSum
            PNetworkSize = NetworkSize 
            
            # If passenger type = Type then reduce PNetworkSum and PNetworkSize by one. This step avoids data 
            # leakage from current passenger similar to how Parch and SibSp are calcualted in the original dataset. 
            if data.loc[index, PtypeField]==1:
                PNetworkSum = PNetworkSum - row['Status']
                PNetworkSize = PNetworkSize - 1

            if PNetworkSize > 0: 
                data.loc[index,NstatusField] = PNetworkSum / PNetworkSize
    data.reset_index(inplace=True)     
TypeList = ['_F_Ch', '_AdultM']
for Type in TypeList:
    update_passenger_network_status(data, Type)
plt.figure(figsize=(12,5))
ax=sns.swarmplot(y='Nstatus_F_Ch', x='Ptype_F_Ch', hue='Survived', data=data[(data.Ptype_F_Ch==1)&(data.InNetwork==1)]);
plt.ylabel('Network Status (Other Women & Children)');
plt.title('Survival of Networked Women & Children');
ax.xaxis.set_visible(False);
plt.figure(figsize=(12,5))
ax=sns.swarmplot(y='Nstatus_AdultM', x='Ptype_AdultM', hue='Survived', data=data[(data.Ptype_AdultM==1)&(data.InNetwork==1)]);
plt.ylabel('Network Status (Other Adult Men)');
plt.title('Survival of Networked Adult Men');
ax.xaxis.set_visible(False);
# split back into training and test datasets for Kaggle competition:
data.set_index('PassengerId', inplace=True)
data_train = data[:train_len]
data_test = data[train_len:]

model_features = ['Ptype_AdultM','Nstatus_F_Ch','Nstatus_AdultM']
X = data_train[model_features].values
y = data_train.Survived.values
SEED=1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

steps = [('tree', DecisionTreeClassifier())]
parameters = {'tree__max_depth': [2],
              'tree__min_samples_leaf': [3,4,5,6,7,8],
              'tree__criterion': ['gini','entropy']}
pipeline = Pipeline(steps)
model = GridSearchCV(pipeline, param_grid=parameters, cv=10)

model.fit(X_train, y_train)
print('Tuned Model Parameters: {}'.format(model.best_params_))
print("Test Set Accuracy: {:.2%}".format(model.score(X_test, y_test)))
# Fit the model
dt = DecisionTreeClassifier(max_depth=2, min_samples_leaf=3, random_state=SEED)
dt = dt.fit(X, y)

# Display Feature Importance
importances = pd.Series(dt.feature_importances_, index=model_features)
sorted_importances = importances.sort_values()
sorted_importances.plot(kind='barh', color='lightgreen', figsize=(6,2));
plt.title('Feature Importance');plt.show()

# Visualize the Decision Tree Classifier
Target_names = ['Died','Survived'] 
dot_data = StringIO()
export_graphviz(dt, out_file=dot_data,
                 feature_names=model_features, 
                 class_names=Target_names,
                 filled=True, rounded=True,
                 special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
results = data_test[model_features].copy()
results['Survived'] = dt.predict(results).astype(int)
results = results[['Survived']]
results.to_csv('DecisionTreeClassifier_engineered_features.csv')