#import necessary libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools
from sklearn.preprocessing import StandardScaler

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, \
confusion_matrix, roc_auc_score

%matplotlib inline
sns.set(font_scale=1.4)
sns.set_style("darkgrid")
# load the data and split the training set into two parts, where the latter part is going to be used
# to determine how well our model is

train_data=pd.read_csv('/kaggle/input/titanic/train.csv')
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
train, validation = np.split(train_data.sample(frac=1, random_state=1003), [int(0.8 * len(train_data))])

data = [train, validation, test_data]
train.shape, validation.shape, test_data.shape
train.head()
train.describe()
for x in data:
    print(x.info(), '\n')
    print(x.isnull().sum(), '\n')
#plotting countplot for the features 'Sex', 'Pclass', 'Embarked', 'Parch' and 'SibSp'

fig, ([ax1, ax2, ax3, ax4, ax5]) = plt.subplots(nrows=1, ncols=5, figsize=(25,5))
fig.tight_layout(pad=2.0)
axes=[ax1, ax2, ax3, ax4, ax5]
data_frame_columns=['Sex', 'Pclass', 'Embarked', 'Parch', 'SibSp']
for x in axes:
    x.grid(False)
for i,j in zip(axes,data_frame_columns):
        sns.countplot(x=j, data=train, palette='deep', ax=i)        
plt.show()
#A function that we will use in several of our computations. It returns three dataframes from any given one.
def fsurvive(df):
    return [df[df.Survived==1], df[df.Survived==0], df]
#Defining the dataframes d_Sex, d_Pclass, d_Embarked. These are the dataframes that we will plot. 

d_Sex=pd.DataFrame(np.array([[len(y[y.Sex==x]) for x in ['male', 'female']] +[len(y)] for y in fsurvive(train)]),\
                   columns=['Male', 'Female', 'Total'], index=['Survived', 'Died', 'Total'])
d_Pclass=pd.DataFrame(np.array([[len(y[y.Pclass==x]) for x in range(1,4)] +[len(y)] for y in fsurvive(train)]),\
                   columns=['Pclass1', 'Pclass2', 'Pclass3', 'Total'], index=['Survived', 'Died', 'Total'])
d_Embarked=pd.DataFrame(np.array([[len(y[y.Embarked==x]) for x in ['S', 'C', 'Q']] +[len(y)] for y in fsurvive(train)]),\
                   columns=['Embarked S', 'Embarked C', 'Embarked Q', 'Total'], index=['Survived', 'Died', 'Total'])
data_frames=[d_Sex, d_Pclass, d_Embarked]
data_frame_columns=['Sex', 'Pclass', 'Embarked']

#plotting

fig, ([ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9]) = plt.subplots(nrows=3, ncols=3, figsize=(25,15))
fig.tight_layout(pad=5.0)
Figure_Titles=['Correlation of Sex and Survival', 'Correlation of Pclass and Survival', 'Correlation of Embarked and Survival', \
              'Ratio of Sex and Survival', 'Ratio of Pclass and Survival', 'Ratio of Embarked and Survival', 'Count Plot - Sex and Survival',\
             'Count Plot - Pclass and Survival', 'Count Plot - Embarked and Survival']
axes=[ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
#adding titles
for x,y in zip(axes,Figure_Titles):
    x.set(title=y)
#closing grids for barchart and countplot
for x in axes[3:]:
    x.grid(False)
#adding heatmaps
for x,y in zip(axes[:3],data_frames):
    sns.heatmap(y, cmap="YlGnBu", annot=True, fmt='d', annot_kws={"size": 20},  cbar=False, ax=x)
    plt.setp(x.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
#adding barplots
for i,j in zip(axes[3:6],data_frame_columns):
        sns.barplot(x=j, y="Survived", data=train, palette="Blues_d", ax=i, ci=None)
#adding countplots
for i,j in zip(axes[6:],data_frame_columns):
        sns.countplot(x="Survived", hue=j, data=train, palette='dark', ax=i)
plt.show()
#Defininf the dataframes d_Parch and d_SibSp. These are the dataframes that we will plot.

d_Parch=pd.DataFrame(np.array([[len(y[y.Parch==x]) for x in range(0,9)] +[len(y)] for y in fsurvive(train)]),\
                   columns=['Parch0', 'Parch1', 'Parch2', 'Parch3', 'Parch4', 'Parch5', 'Parch6', 'Parch7', 'Parch8', 'Total'], \
                     index=['Survived', 'Died', 'Total'])
d_SibSp=pd.DataFrame(np.array([[len(y[y.SibSp==x]) for x in range(0,9)] +[len(y)] for y in fsurvive(train)]),\
                   columns=['SibSp0', 'SibSp1', 'SibSp2', 'SibSp3', 'SibSp4', 'SibSp5', 'SibSp6', 'SibSp7', 'SibSp8', 'Total'], \
                     index=['Survived', 'Died', 'Total'])
data_frames=[d_Parch, d_SibSp]
data_frame_columns=['Parch', 'SibSp']

#plotting

fig, ([ax1, ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(nrows=3, ncols=2, figsize=(25,15))
fig.tight_layout(pad=5.0)
Figure_Titles=['Correlation of Parch and Survival', 'Correlation of SibSp and Survival', 'Ratio of Parch and Survival', \
               'Ratio of SibSp and Survival', 'Count Plot - Parch and Survival', 'Count Plot - SibSp and Survival']
axes=[ax1, ax2, ax3, ax4, ax5, ax6]
#adding titles
for x,y in zip(axes,Figure_Titles):
    x.set(title=y)
#closing grids for barchart and countplot
for x in axes[2:]:
    x.grid(False)
#adding heatmaps
for x,y in zip(axes[:2],data_frames):
    sns.heatmap(y, cmap="YlGnBu", annot=True, fmt='d', annot_kws={"size": 20},  cbar=False, ax=x)
    plt.setp(x.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
#adding barplots
for i,j in zip(axes[2:4],data_frame_columns):
        sns.barplot(x=j, y="Survived", data=train, palette="rocket", ax=i, ci=None)
#adding countplots
for i,j in zip(axes[4:],data_frame_columns):
        sns.countplot(x="Survived", hue=j, data=train, palette='deep', ax=i)
        i.legend(loc='upper right')
plt.show()
#We will plot 4 dataframes: 1) data that miss the Cabin and Age, 2) data that miss the Cabin but not Age info 
#3) data that miss Age but not Cabin 4) data that do not miss neither Cabin or Age

#CA_list is the list of the 4 dataframes that we will plot
CA_list  = [train[train.Cabin.isnull() & train.Age.isnull()], train[train.Cabin.isnull() & train.Age.notnull()], \
train[train.Cabin.notnull() & train.Age.isnull()], train[train.Cabin.notnull() & train.Age.notnull()]]

#plotting
fig, ([ax1, ax2, ax3, ax4]) = plt.subplots(nrows=1, ncols=4, figsize=(25,5))
Figure_Titles=['Missing Both (Cabin) and (Age)', 'Missing (Cabin) but not (Age)', \
               'Missing (Age) but not (Cabin)', 'Not missing either of (Cabin) or (Age)']
for k,l,m,s in zip(CA_list,[ax1,ax2,ax3,ax4],[True, False, False, False], Figure_Titles):
    d=pd.DataFrame(np.array([[len(y[y.Sex==x]) for x in ['male', 'female']] +[len(y)] for y in fsurvive(k)]),\
                   columns=['Male', 'Female', 'Total'], index=['Survived', 'Died', 'Total'])
    sns.heatmap(d, cmap="Spectral", annot=True, fmt='d', annot_kws={"size": 20},  cbar=False, ax=l, yticklabels=m)
    l.set(title = s)
plt.setp(ax1.get_yticklabels(), rotation=0, ha="right",
         rotation_mode="anchor")
plt.show()

#We will plot 12 dataframes. CA_list was defined in the previous cell.

fig, ([Qax1, Qax2, Sax1, Sax2, Cax1, Cax2], [Qax3, Qax4, Sax3, Sax4, Cax3, Cax4]) = \
plt.subplots(nrows=2, ncols=6, figsize=(30,10))

Figure_Titles=['C-A- | EmQ', 'C-A+ | EmQ', 'C+A- | EmQ', 'C+A+ | EmQ', 'C-A- | EmS', 'C-A+ | EmS', 'C+A- | EmS', 'C+A+ | EmS', \
              'C-A- | EmC', 'C-A+ | EmC', 'C+A- | EmC', 'C+A+ | EmC']

#plotting
for k, l, m, s in zip([x[x.Embarked==k] for k,x in itertools.product(['Q', 'S', 'C'], CA_list)], \
                [Qax1, Qax2, Qax3, Qax4, Sax1, Sax2, Sax3, Sax4, Cax1, Cax2, Cax3, Cax4], \
        [True, False, True, False, False, False, False, False, False, False, False, False], Figure_Titles):
    d=pd.DataFrame(np.array([[len(y[y.Sex==x]) for x in ['male', 'female']] +[len(y)] for y in fsurvive(k)]),\
                    columns=['Male', 'Female', 'Total'], index=['Survived', 'Died', 'Total'])
    sns.heatmap(d, cmap="Spectral", annot=True, fmt='d', annot_kws={"size": 20},  cbar=False, ax=l, yticklabels=m)
    l.set(title = s)
#design
plt.setp(Qax1.get_yticklabels(), rotation=0, ha="right",
         rotation_mode="anchor")
plt.setp(Qax3.get_yticklabels(), rotation=0, ha="right",
         rotation_mode="anchor")
fig.subplots_adjust(hspace=.4)
plt.show()
#We will plot 12 dataframes. CA_list was defined earlier. the notation prefixes Q,S,C are stands for Pclass1, Pclass2, Pclass3, respectively.

fig, ([Qax1, Qax2, Sax1, Sax2, Cax1, Cax2], [Qax3, Qax4, Sax3, Sax4, Cax3, Cax4]) = \
plt.subplots(nrows=2, ncols=6, figsize=(30,10))

Figure_Titles=['C-A- | Pclass=1', 'C-A+ | Pclass=1', 'C+A- | Pclass=1', 'C+A+ | Pclass=1', 'C-A- | Pclass=2', 'C-A+ | Pclass=2', \
               'C+A- | Pclass=2', 'C+A+ | Pclass=2', 'C-A- | Pclass=3', 'C-A+ | Pclass=3', 'C+A- | Pclass=3', 'C+A+ | Pclass=3']

#plotting
for k, l, m, s in zip([x[x.Pclass==k] for k,x in itertools.product([1,2,3], CA_list)], \
                [Qax1, Qax2, Qax3, Qax4, Sax1, Sax2, Sax3, Sax4, Cax1, Cax2, Cax3, Cax4], \
        [True, False, True, False, False, False, False, False, False, False, False, False], Figure_Titles):
    d=pd.DataFrame(np.array([[len(y[y.Sex==x]) for x in ['male', 'female']] +[len(y)] for y in fsurvive(k)]),\
                    columns=['Male', 'Female', 'Total'], index=['Survived', 'Died', 'Total'])
    sns.heatmap(d, cmap="Spectral", annot=True, fmt='d', annot_kws={"size": 20},  cbar=False, ax=l, yticklabels=m)
    l.set(title = s)
#design
plt.setp(Qax1.get_yticklabels(), rotation=0, ha="right",
         rotation_mode="anchor")
plt.setp(Qax3.get_yticklabels(), rotation=0, ha="right",
         rotation_mode="anchor")
fig.subplots_adjust(hspace=.4)
plt.show()
Unique_Tickets_train=train.Ticket.unique().tolist()
Unique_Tickets_validation=validation.Ticket.unique().tolist()
Unique_Tickets_test=test_data.Ticket.unique().tolist()

List_of_Words_train=[]
List_of_Words_validation=[]
List_of_Words_test=[]
for x in Unique_Tickets_train:
    if any(c.isalpha() for c in x)==True:
        List_of_Words_train.append(x)

for x in Unique_Tickets_validation:
    if any(c.isalpha() for c in x)==True:
        List_of_Words_validation.append(x)
        
for x in Unique_Tickets_test:
    if any(c.isalpha() for c in x)==True:
        List_of_Words_test.append(x)

####train
Tickets_SC = [s for s in Unique_Tickets_train if ("S.C." in s) or ("SC" in s) or ("SC." in s) or ("S.C" in s)]
Tickets_PC = [s for s in Unique_Tickets_train if ("P.C." in s) or ("PC" in s) or ("PC." in s) or ("P.C" in s)]
Tickets_STON = [s for s in Unique_Tickets_train if ("SOTON" in s) or ("STON" in s)]
Tickets_CA = [s for s in Unique_Tickets_train if ("C.A. " in s) or ("CA. " in s) or \
              ("C.A. " in s) or ("C.A " in s) or ('CA ' in s)]
Tickets_LP=[s for s in Unique_Tickets_train if 'LP 1588' in s]
Tickets_A = [s for s in Unique_Tickets_train if ("A.5" in s) or ("A/" in s) or ("A./5" in s) or ("A./4" in s) or ("A4." in s)\
            or ('AQ/4 3130' in s) or ('A. 2. 39186' in s) or ('AQ/3. 30631' in s)]
Tickets_SO = [s for s in Unique_Tickets_train if ("S.O" in s) or ('SO/' in s)]
Tickets_P = [s for s in Unique_Tickets_train if ("P/" in s) or ('PP ' in s)]
Tickets_C = [s for s in Unique_Tickets_train if ("C 7075" in s) or ("C 7076" in s) or \
             ('C 17369' in s) or ('C 7077' in s) or ('C 17368' in s) or ('C 4001' in s)]
Tickets_Line = [s for s in Unique_Tickets_train if "LINE" in s ]
Tickets_F = [s for s in Unique_Tickets_train if ("F.C." in s) or ('Fa' in s)]
Tickets_WE = [s for s in Unique_Tickets_train if ("WE" in s) or ("W/C" in s) or ('W./C' in s) or ('W.E.P.' in s)]
Tickets_SP = [s for s in Unique_Tickets_train if "S.P. 3464" in s ]

####validation 

v_Tickets_SC = [s for s in Unique_Tickets_validation if ("S.C." in s) or ("SC" in s) or ("SC." in s) or ("S.C" in s)]
v_Tickets_PC = [s for s in Unique_Tickets_validation if ("P.C." in s) or ("PC" in s) or ("PC." in s) or ("P.C" in s)]
v_Tickets_STON = [s for s in Unique_Tickets_validation if ("SOTON" in s) or ("STON" in s)]
v_Tickets_CA = [s for s in Unique_Tickets_validation if ("C.A. " in s) or ("CA. " in s) or \
              ("C.A. " in s) or ("C.A " in s) or ('CA ' in s)]
v_Tickets_LP=[s for s in Unique_Tickets_validation if 'LP 1588' in s]
v_Tickets_A = [s for s in Unique_Tickets_validation if ("A.5" in s) or ("A/" in s) or ("A./5" in s) or ("A./4" in s) or ("A4." in s)\
            or ('AQ/4 3130' in s) or ('A. 2. 39186' in s) or ('AQ/3. 30631' in s)]
v_Tickets_SO = [s for s in Unique_Tickets_validation if ("S.O" in s) or ('SO/' in s)]
v_Tickets_P = [s for s in Unique_Tickets_validation if ("P/" in s) or ('PP ' in s)]
v_Tickets_C = [s for s in Unique_Tickets_validation if ("C 7075" in s) or ("C 7076" in s) or \
             ('C 17369' in s) or ('C 7077' in s) or ('C 17368' in s) or ('C 4001' in s)]
v_Tickets_Line = [s for s in Unique_Tickets_validation if ("LINE" in s) ]
v_Tickets_F = [s for s in Unique_Tickets_validation if ("F.C." in s) or ('Fa' in s)]
v_Tickets_WE = [s for s in Unique_Tickets_validation if ("WE" in s) or ("W/C" in s) or ('W./C' in s) or ('W.E.P.' in s)]
v_Tickets_SP = [s for s in Unique_Tickets_validation if ("S.P. 3464" in s) ]

###test


t_Tickets_SC = [s for s in Unique_Tickets_test if ("S.C." in s) or ("SC" in s) or ("SC." in s) or ("S.C" in s)]
t_Tickets_PC = [s for s in Unique_Tickets_test if ("P.C." in s) or ("PC" in s) or ("PC." in s) or ("P.C" in s)]
t_Tickets_STON = [s for s in Unique_Tickets_test if ("SOTON" in s) or ("STON" in s)]
t_Tickets_CA = [s for s in Unique_Tickets_test if ("C.A." in s) or ("CA." in s) or \
              ("C.A." in s) or ("C.A" in s) or ('CA' in s)]
t_Tickets_LP=[s for s in Unique_Tickets_test if 'LP 1588' in s]
t_Tickets_A = [s for s in Unique_Tickets_test if ("A.5" in s) or ("A/" in s) or ("A./" in s) or ("A4." in s)\
            or ('AQ/4 3130' in s) or ('A. 2. 39186' in s) or ('AQ/3. 30631' in s)]
t_Tickets_SO = [s for s in Unique_Tickets_test if ("S.O" in s) or ('SO/' in s)]
t_Tickets_P = [s for s in Unique_Tickets_test if ("P/" in s) or ('PP ' in s)]
t_Tickets_C = [s for s in Unique_Tickets_test if ("C 7075" in s) or ("C 7076" in s) or \
             ('C 17369' in s) or ('C 7077' in s) or ('C 17368' in s) or ('C 4001' in s)]
t_Tickets_Line = [s for s in Unique_Tickets_test if ("LINE" in s) ]
t_Tickets_F = [s for s in Unique_Tickets_test if ("F.C." in s) or ('Fa' in s)]
t_Tickets_WE = [s for s in Unique_Tickets_test if ("WE" in s) or ("W/C" in s) or ('W./C' in s) or ('W.E.P.' in s)]
t_Tickets_SP = [s for s in Unique_Tickets_test if ("S.P. 3464" in s) ]

Totall_train=Tickets_SP + Tickets_WE + Tickets_F + Tickets_Line + Tickets_C + Tickets_P + Tickets_SO + Tickets_A + Tickets_LP +\
Tickets_CA + Tickets_STON + Tickets_PC + Tickets_SC 

Totall_validation=v_Tickets_SP +v_Tickets_WE + v_Tickets_F + v_Tickets_Line + v_Tickets_C + v_Tickets_P + v_Tickets_SO + v_Tickets_A + v_Tickets_LP +\
v_Tickets_CA + v_Tickets_STON + v_Tickets_PC + v_Tickets_SC 

Totall_test=t_Tickets_SP + t_Tickets_WE + t_Tickets_F + t_Tickets_Line + t_Tickets_C + t_Tickets_P + t_Tickets_SO + t_Tickets_A + t_Tickets_LP +\
t_Tickets_CA + t_Tickets_STON + t_Tickets_PC + t_Tickets_SC 


print(len(List_of_Words_train), len(Totall_train), '\n')
print(len(List_of_Words_validation), len(Totall_validation), '\n')
print(len(List_of_Words_test), len(Totall_test), '\n')
Totall_validation=v_Tickets_SP +v_Tickets_WE + v_Tickets_F + v_Tickets_Line + v_Tickets_C + v_Tickets_P + v_Tickets_SO + v_Tickets_A + v_Tickets_LP +\
v_Tickets_CA + v_Tickets_STON + v_Tickets_PC + v_Tickets_SC 

if (len(Totall_validation) == len(set(Totall_validation))) & (len(Totall_train) == len(set(Totall_train))) & (len(Totall_test) == len(set(Totall_test))):
    print('yay')
else:
    print('ay')
Tickets_SP + Tickets_WE + Tickets_F + Tickets_Line + Tickets_C + Tickets_P + Tickets_SO + Tickets_A + Tickets_LP +\
Tickets_CA + Tickets_STON + Tickets_PC + Tickets_SC 
for x in data:
    x['Ticket_Prefix']=0


train.loc[(train['Ticket'].isin(Tickets_SP)) , 'Ticket_Prefix']=1
train.loc[(train['Ticket'].isin(Tickets_WE)) , 'Ticket_Prefix']=2
train.loc[(train['Ticket'].isin(Tickets_F)) , 'Ticket_Prefix']=3
train.loc[(train['Ticket'].isin(Tickets_C)) , 'Ticket_Prefix']=4
train.loc[(train['Ticket'].isin(Tickets_P)) , 'Ticket_Prefix']=5
train.loc[(train['Ticket'].isin(Tickets_SO)) , 'Ticket_Prefix']=6
train.loc[(train['Ticket'].isin(Tickets_A)) , 'Ticket_Prefix']=7
train.loc[(train['Ticket'].isin(Tickets_LP)) , 'Ticket_Prefix']=8
train.loc[(train['Ticket'].isin(Tickets_CA)) , 'Ticket_Prefix']=9
train.loc[(train['Ticket'].isin(Tickets_STON)) , 'Ticket_Prefix']=10
train.loc[(train['Ticket'].isin(Tickets_PC)) , 'Ticket_Prefix']=11
train.loc[(train['Ticket'].isin(Tickets_SC)) , 'Ticket_Prefix']=12

validation.loc[(validation['Ticket'].isin(v_Tickets_SP)) , 'Ticket_Prefix']=1
validation.loc[(validation['Ticket'].isin(v_Tickets_WE)) , 'Ticket_Prefix']=2
validation.loc[(validation['Ticket'].isin(v_Tickets_F)) , 'Ticket_Prefix']=3
validation.loc[(validation['Ticket'].isin(v_Tickets_C)) , 'Ticket_Prefix']=4
validation.loc[(validation['Ticket'].isin(v_Tickets_P)) , 'Ticket_Prefix']=5
validation.loc[(validation['Ticket'].isin(v_Tickets_SO)) , 'Ticket_Prefix']=6
validation.loc[(validation['Ticket'].isin(v_Tickets_A)) , 'Ticket_Prefix']=7
validation.loc[(validation['Ticket'].isin(v_Tickets_LP)) , 'Ticket_Prefix']=8
validation.loc[(validation['Ticket'].isin(v_Tickets_CA)) , 'Ticket_Prefix']=9
validation.loc[(validation['Ticket'].isin(v_Tickets_STON)) , 'Ticket_Prefix']=10
validation.loc[(validation['Ticket'].isin(v_Tickets_PC)) , 'Ticket_Prefix']=11
validation.loc[(validation['Ticket'].isin(v_Tickets_SC)) , 'Ticket_Prefix']=12


test_data.loc[(test_data['Ticket'].isin(t_Tickets_SP)) , 'Ticket_Prefix']=1
test_data.loc[(test_data['Ticket'].isin(t_Tickets_WE)) , 'Ticket_Prefix']=2
test_data.loc[(test_data['Ticket'].isin(t_Tickets_F)) , 'Ticket_Prefix']=3
test_data.loc[(test_data['Ticket'].isin(t_Tickets_C)) , 'Ticket_Prefix']=4
test_data.loc[(test_data['Ticket'].isin(t_Tickets_P)) , 'Ticket_Prefix']=5
test_data.loc[(test_data['Ticket'].isin(t_Tickets_SO)) , 'Ticket_Prefix']=6
test_data.loc[(test_data['Ticket'].isin(t_Tickets_A)) , 'Ticket_Prefix']=7
test_data.loc[(test_data['Ticket'].isin(t_Tickets_LP)) , 'Ticket_Prefix']=8
test_data.loc[(test_data['Ticket'].isin(t_Tickets_CA)) , 'Ticket_Prefix']=9
test_data.loc[(test_data['Ticket'].isin(t_Tickets_STON)) , 'Ticket_Prefix']=10
test_data.loc[(test_data['Ticket'].isin(t_Tickets_PC)) , 'Ticket_Prefix']=11
test_data.loc[(test_data['Ticket'].isin(t_Tickets_SC)) , 'Ticket_Prefix']=12



train['Ticket_Prefix'].unique(), validation['Ticket_Prefix'].unique(), test_data['Ticket_Prefix'].unique()
asubdf1=train[(train.Age.notnull()) & (train.Cabin.isnull()) & (train['Sex']=='male')]
asubdf2=train[(train.Age.notnull()) & (train.Cabin.notnull()) & (train['Sex']=='male')]
asubdf3=train[(train.Age.notnull()) & (train.Cabin.isnull()) & (train['Sex']=='female')]
asubdf4=train[(train.Age.notnull()) & (train.Cabin.notnull()) & (train['Sex']=='female')]


av_subdf1=validation[(validation.Age.notnull()) & (validation.Cabin.isnull()) & (validation['Sex']=='male')]
av_subdf2=validation[(validation.Age.notnull()) & (validation.Cabin.notnull()) & (validation['Sex']=='male')]
av_subdf3=validation[(validation.Age.notnull()) & (validation.Cabin.isnull()) & (validation['Sex']=='female')]
av_subdf4=validation[(validation.Age.notnull()) & (validation.Cabin.notnull()) & (validation['Sex']=='female')]

at_subdf1=test_data[(test_data.Age.notnull()) & (test_data.Cabin.isnull()) & (test_data['Sex']=='male')]
at_subdf2=test_data[(test_data.Age.notnull()) & (test_data.Cabin.notnull()) & (test_data['Sex']=='male')]
at_subdf3=test_data[(test_data.Age.notnull()) & (test_data.Cabin.isnull()) & (test_data['Sex']=='female')]
at_subdf4=test_data[(test_data.Age.notnull()) & (test_data.Cabin.notnull()) & (test_data['Sex']=='female')]

for x in [train, validation, test_data]:
    x.loc[(x.Cabin.isnull()), 'Cabin']='Unknown'
    x['CInit']=x['Cabin'].str.split('',expand=True)[1]
train.CInit.unique()

validation.CInit.unique()
#train.loc[(train['CInit']=='T'),'CInit']='C'
validation.loc[(validation['CInit']=='T'),'CInit']='C'
train.loc[(train['CInit']=='G'),'CInit']='C'
#validation.loc[(validation['CInit']=='G'),'CInit']='C'
test_data.loc[(test_data['CInit']=='G'),'CInit']='C'

train_dum2=pd.get_dummies(train['CInit'], prefix='Cabin_')
train=pd.concat([train, train_dum2], axis=1)

validation_dum2=pd.get_dummies(validation['CInit'], prefix='Cabin_')
validation=pd.concat([validation, validation_dum2], axis=1)

test_dum2=pd.get_dummies(test_data['CInit'], prefix='Cabin_')
test_data=pd.concat([test_data, test_dum2], axis=1)

validation.loc[(validation['Embarked'].isnull()),'Embarked']='S'
train.loc[(train['Embarked'].isnull()),'Embarked']='S'
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
data = [train, validation, test_data]




##################

for x in data:
    x["NullCabin"] = 1
    x["NullAge"] = 1
    x['TicketDuplicate']=0
    x["TicketNumber"] = x['Ticket'].str.split(' ').str[-1]
    x["SP"] = 0
    x["S0P0"] = 0
    x["S0P1"] = 0
    x["S1P0"] = 0
    x["S1P1"] = 0
    x["S2P0"] = 0
    x["S0P2"] = 0
    x["S1P1"] = 0
    x["SP_rest"] = 0
    x["C_A_E"] = 0
    x["C-A-EmQ"] = 0
    x["C-A+EmQ"] = 0
    x["C+A-EmQ"] = 0
    x["C+A+EmQ"] = 0
    x["C-A-EmS"] = 0
    x["C-A+EmS"] = 0
    x["C+A-EmS"] = 0
    x["C+A+EmS"] = 0
    x["C-A-EmC"] = 0
    x["C-A+EmC"] = 0
    x["C+A-EmC"] = 0
    x["C+A+EmC"] = 0
    x.loc[((x.Cabin=='Unknown') & (x.Age.isnull()) & (x["Embarked"]=='Q')), 'C-A-EmQ']=1
    x.loc[((x.Cabin=='Unknown') & (x.Age.isnull()) & (x["Embarked"]=='S')), 'C-A-EmS']=1
    x.loc[((x.Cabin=='Unknown') & (x.Age.isnull()) & (x["Embarked"]=='C')), 'C-A-EmC']=1
    x.loc[((x.Cabin=='Unknown') & (x.Age.notnull()) & (x["Embarked"]=='Q')), 'C-A+EmQ']=1
    x.loc[((x.Cabin=='Unknown') & (x.Age.notnull()) & (x["Embarked"]=='S')), 'C-A+EmS']=1
    x.loc[((x.Cabin=='Unknown') & (x.Age.notnull()) & (x["Embarked"]=='C')), 'C-A+EmC']=1
    x.loc[((x.Cabin!='Unknown') & (x.Age.isnull()) & (x["Embarked"]=='Q')), 'C+A-EmQ']=1
    x.loc[((x.Cabin!='Unknown') & (x.Age.isnull()) & (x["Embarked"]=='S')), 'C+A-EmS']=1
    x.loc[((x.Cabin!='Unknown') & (x.Age.isnull()) & (x["Embarked"]=='C')), 'C+A-EmC']=1
    x.loc[((x.Cabin!='Unknown') & (x.Age.notnull()) & (x["Embarked"]=='Q')), 'C+A+EmQ']=1
    x.loc[((x.Cabin!='Unknown') & (x.Age.notnull()) & (x["Embarked"]=='S')), 'C+A+EmS']=1
    x.loc[((x.Cabin!='Unknown') & (x.Age.notnull()) & (x["Embarked"]=='C')), 'C+A+EmC']=1
    x.loc[((x.Cabin.notnull()) & (x.Age.isnull()) & (x["Embarked"]=='Q')), 'C_A_E']=7
    x.loc[((x.Cabin.notnull()) & (x.Age.isnull()) & (x["Embarked"]=='S')), 'C_A_E']=8
    x.loc[((x.Cabin.notnull()) & (x.Age.isnull()) & (x["Embarked"]=='C')), 'C_A_E']=9
    x.loc[((x.Cabin.notnull()) & (x.Age.notnull()) & (x["Embarked"]=='Q')), 'C_A_E']=10
    x.loc[((x.Cabin.notnull()) & (x.Age.notnull()) & (x["Embarked"]=='S')), 'C_A_E']=11
    x.loc[((x.Cabin.notnull()) & (x.Age.notnull()) & (x["Embarked"]=='C')), 'C_A_E']=12
    x.loc[((x["SibSp"]==0) & (x["Parch"]==0)), 'S0P0']=1
    x.loc[((x["SibSp"]==0) & (x["Parch"]==1)), 'S0P1']=1
    x.loc[((x["SibSp"]==1) & (x["Parch"]==0)), 'S1P0']=1
    x.loc[((x["SibSp"]==1) & (x["Parch"]==1)), 'S1P1']=1 
    x.loc[((x["SibSp"]==0) & (x["Parch"]==2)), 'S0P2']=1
    x.loc[((x["SibSp"]==2) & (x["Parch"]==0)), 'S2P0']=1
    x.loc[((x["SibSp"]==1) & (x["Parch"]==1)), 'S1P1']=1
    x.loc[((x["SibSp"]+x["Parch"]>2)), 'SP_rest']=1
    x.loc[((x["SibSp"]==0) & (x["Parch"]==0)), 'SP']=1
    x.loc[((x["SibSp"]==0) & (x["Parch"]==1)), 'SP']=2
    x.loc[((x["SibSp"]==1) & (x["Parch"]==0)), 'SP']=3
    x.loc[((x["SibSp"]==1) & (x["Parch"]==1)), 'SP']=4
    x.loc[((x["SibSp"]==0) & (x["Parch"]==2)), 'SP']=5
    x.loc[((x["SibSp"]==2) & (x["Parch"]==0)), 'SP']=6
    x.loc[((x["SibSp"]==1) & (x["Parch"]==1)), 'SP']=7
    x.loc[x.duplicated(subset=['Ticket'], keep=False), 'TicketDuplicate']=1
    x.loc[(x["Cabin"]=='Unknown'), 'NullCabin']=0
    x.loc[(x["Age"].notnull()), 'NullAge']=0


train.loc[(train.TicketNumber=='LINE'), 'TicketNumber'] = 0
validation.loc[(validation.TicketNumber=='LINE'), 'TicketNumber'] = 0

train.TicketNumber=train['TicketNumber'].astype('int') 
validation.TicketNumber=validation['TicketNumber'].astype('int') 
test_data.TicketNumber=test_data['TicketNumber'].astype('int') 

data = [train, validation, test_data] 

for x in [train, test_data]:
    x['Cabin__A']=x['Cabin__A'].astype('int')
    x['Cabin__B']=x['Cabin__B'].astype('int')
    x['Cabin__C']=x['Cabin__C'].astype('int')
    x['Cabin__D']=x['Cabin__D'].astype('int')
    x['Cabin__E']=x['Cabin__E'].astype('int')
    x['Cabin__U']=x['Cabin__U'].astype('int')
    x['Cabin__F']=x['Cabin__F'].astype('int')
    

#train['Cabin__D']=train['Cabin__D'].astype('int')
#test_data['Cabin__D']=test_data['Cabin__D'].astype('int')
#validation['Cabin__D']=0

data = [train, validation, test_data]  


Gender = ['male', 'female']
for x in data:
    for i in [0,1]:
        x.loc[(x['Sex']==Gender[i]), 'Sex']=i
    x['Sex']=x['Sex'].astype('int')

data = [train, validation, test_data]
####### Change Embarked Feature Next #####
Embarked_list = ['Q', 'S', 'C']
for x in data:
    for j in [0, 1, 2]:
        x.loc[(x['Embarked'] == Embarked_list[j]),'Embarked']=j 
    x['Embarked']=x['Embarked'].astype('int')

data = [train, validation, test_data]






##### NAME


for x in data:
    x['Title'] = x['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

title_to_num = {"Master": 1, "Miss": 2, "Mr": 3, "Mrs": 4, "Other": 5}
for x in data:
    x.loc[(x['Title'] != 'Mr') & (x['Title'] != 'Master') & (x['Title'] != 'Mrs') & \
          (x['Title'] != 'Miss') , 'Title'] = 'Other'
    x['Title'] = x['Title'].map(title_to_num)

data = [train, validation, test_data]



Mr_list=[399, 767, 151, 627, 150, 600, 661, 648, 695, 31, \
 250, 633, 450, 537, 318, 849, 746, 823, 887, 246, 1023, 1041, 1056, 1094, 1185]
    
Miss_list=[760, 444, 797, 370, 557, 642, 980, 1306]

Mrs_list=[711]

Id_distribution = []
for x in data:
    a=x[x['Title']==5].PassengerId
    Id_distribution.append(a)

for x in data:
    for i in [0,1,2]:
        for y in Id_distribution[i]:
            if y in Mr_list:
                x.loc[(x['PassengerId']==y), 'Title'] = 3
            elif y in Miss_list:
                x.loc[(x['PassengerId']==y), 'Title'] = 2
            elif y in Mrs_list:
                x.loc[(x['PassengerId']==y), 'Title'] = 4
    x['Title']=x['Title'].astype('int')

data = [train, validation, test_data]

###  
for x in data:
    x['Mr']=0
    x['MMr']=0
    x['MMMr']=0
    x['Miss']=0
    x['Mrs']=0
    x['Master']=0
    x.loc[(x.Title==3), 'Mr']=1
    x.loc[(x.Title==4), 'Mrs']=1
    x.loc[(x.Title==2), 'Miss']=1
    x.loc[(x.Title==1), 'Master']=1
    x['FareLog']=np.log1p(x['Fare'])
data = [train, validation, test_data]

###########
train.loc[((train['Age'].isnull()) & (train['Sex'] == 0) & (train['CInit']=='U')), 'Age'] = asubdf1.Age.median()
train.loc[((train['Age'].isnull()) & (train['Sex'] == 0) & (train['CInit']!='U')), 'Age'] = asubdf2.Age.median()
train.loc[((train['Age'].isnull()) & (train['Sex'] == 1) & (train['CInit']=='U')), 'Age'] = asubdf3.Age.median()
train.loc[((train['Age'].isnull()) & (train['Sex'] == 1) & (train['CInit']!='U')), 'Age'] = asubdf4.Age.median()

validation.loc[((validation['Age'].isnull()) & (validation['Sex'] == 0) & (validation['CInit']=='U')), 'Age'] = av_subdf1.Age.median()
validation.loc[((validation['Age'].isnull()) & (validation['Sex'] == 0) & (validation['CInit']!='U')), 'Age'] = av_subdf2.Age.median()
validation.loc[((validation['Age'].isnull()) & (validation['Sex'] == 1) & (validation['CInit']=='U')), 'Age'] = av_subdf3.Age.median()
validation.loc[((validation['Age'].isnull()) & (validation['Sex'] == 1) & (validation['CInit']!='U')), 'Age'] = av_subdf4.Age.median()


test_data.loc[((test_data['Age'].isnull()) & (test_data['Sex'] == 0) & (test_data['CInit']=='U')), 'Age'] = at_subdf1.Age.median()
test_data.loc[((test_data['Age'].isnull()) & (test_data['Sex'] == 0) & (test_data['CInit']!='U')), 'Age'] = at_subdf2.Age.median()
test_data.loc[((test_data['Age'].isnull()) & (test_data['Sex'] == 1) & (test_data['CInit']=='U')), 'Age'] = at_subdf3.Age.median()
test_data.loc[((test_data['Age'].isnull()) & (test_data['Sex'] == 1) & (test_data['CInit']!='U')), 'Age'] = at_subdf4.Age.median()


data = [train, validation, test_data]

for x in data:
    x['AgeGrp']=0
    x.loc[((x.Age>=0) & (x.Age<10)), 'AgeGrp']=1
    x.loc[((x.Age>=10) & (x.Age<30)), 'AgeGrp']=2
    x.loc[((x.Age>=30) & (x.Age<60)), 'AgeGrp']=3
    x.loc[(x.Age>60), 'AgeGrp']=4

train.CInit.value_counts()
CInit_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}

for x in data:
    x['CInit'] = x['CInit'].map(CInit_map)
    x['CInit']=x['CInit'].astype('int') 
data = [train, validation, test_data]  

for x in data:
    x['TicketNumber'] = (x['TicketNumber']-x.TicketNumber.mean())/(x.TicketNumber.max()-x.TicketNumber.min())
    x['Age'] = (x['Age']-x.Age.mean())/(x.Age.max()-x.Age.min())
    x['FFamS'] = 0
    x.loc[((x.Parch==0) & (x.SibSp==0)), 'FFamS']=1
    x.loc[((x.Parch<=2) & (x.SibSp<=2) & (x.Parch>0) & (x.SibSp>0)), 'FFamS']=2
    x['AclassF']=0
    x.loc[((x.Pclass<3) & (x.Sex==1)), 'AclassF']=1
    x.loc[((x.Pclass>1) & (x.Sex==0)), 'AclassF']=2
    x.loc[((x.Pclass==1) & (x.Sex==0)), 'AclassF']=3

data = [train, validation, test_data]  
Features=['Age', 'Sex', 'Mr',   \
         'FFamS', \
          'C_A_E',\
          'TicketNumber', 'FareLog', 'CInit', 'AclassF']

Features2=['Age', 'Sex', 'Mr',   \
         'FFamS', \
          'C_A_E',\
          'TicketNumber', 'FareLog', 'CInit', 'AclassF']

X_train = train[Features]
Y_train = train["Survived"]
X_validation = validation[Features]
Y_validation = validation["Survived"]

X_train2 = train[Features2]
Y_train2 = train["Survived"]
X_validation2 = validation[Features2]
Y_validation2 = validation["Survived"]


Results_Train = []
Results_Validation = []
Results_Accuracy = []
Results_Precision = []
Results_Recall = []
Results_F1 = []
Results_AUC = []
Results_Confusion = []


Results_Train2 = []
Results_Validation2 = []
Results_Accuracy2 = []
Results_Precision2 = []
Results_Recall2 = []
Results_F12 = []
Results_AUC2 = []
Results_Confusion2 = []



A_random_forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=1093)
                                        

A_random_forest.fit(X_train, Y_train)
RF= round(A_random_forest.score(X_train, Y_train) * 100, 2) 
Results_Train.append(RF)
RF2 = round(A_random_forest.score(X_validation, Y_validation) * 100, 2) 
Results_Validation.append(RF2)
    
Predictions = A_random_forest.predict(X_validation)

accuracy = accuracy_score(Y_validation,Predictions)
precision =precision_score(Y_validation, Predictions)
recall =  recall_score(Y_validation, Predictions)
f1 = f1_score(Y_validation,Predictions)
auc = roc_auc_score(Y_validation,Predictions)
cm = confusion_matrix(Y_validation, Predictions)
    
Results_Accuracy.append(accuracy)
Results_Precision.append(precision)
Results_Recall.append(recall)
Results_F1.append(f1)
Results_AUC.append(auc)
Results_Confusion.append(cm)


daa = {'Results_Train': Results_Train, 'Results_Validation': Results_Validation, \
       'Results_Accuracy': Results_Accuracy, 'Results_Precision': Results_Precision, \
       'Results_Recall': Results_Recall, 'Results_F1': Results_F1, 'Results_AUC': Results_AUC, \
       'Results_Confusion1': Results_Confusion}
df = pd.DataFrame(data=daa)

df


def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


m = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=1453)
m.fit(X_validation, Y_validation)

df=X_validation

fimp =rf_feat_importance(m, df)
fimp.plot('cols', 'imp', 'barh', figsize=(20,10), legend=False) 
plt.grid(False)
for i in range(500,713):
    A_random_forest2 = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=i)#,\
    A_random_forest2.fit(X_train2[:i], Y_train2[:i])
    RF__2= round(A_random_forest2.score(X_train2[:i], Y_train2[:i]) * 100, 2) 
    Results_Train2.append(RF__2)
    RF2__2 = round(A_random_forest2.score(X_validation2, Y_validation2) * 100, 2) 
    Results_Validation2.append(RF2__2)

    Predictions2 = A_random_forest2.predict(X_validation2)

    accuracy2 = accuracy_score(Y_validation2,Predictions2)
    precision2 =precision_score(Y_validation2, Predictions2)
    recall2 =  recall_score(Y_validation2, Predictions2)
    f12 = f1_score(Y_validation2,Predictions2)
    auc2 = roc_auc_score(Y_validation2,Predictions2)
    cm2 = confusion_matrix(Y_validation2, Predictions2)

    Results_Accuracy2.append(accuracy2)
    Results_Precision2.append(precision2)
    Results_Recall2.append(recall2)
    Results_F12.append(f12)
    Results_AUC2.append(auc2)
    Results_Confusion2.append(cm2)


daa2 = {'Results_Train': Results_Train2, 'Results_Validation': Results_Validation2, \
            'Results_Accuracy': Results_Accuracy2, 'Results_Precision': Results_Precision2, \
            'Results_Recall': Results_Recall2, 'Results_F1': Results_F12, 'Results_AUC': Results_AUC2, \
            'Results_Confusion1': Results_Confusion2}
df2 = pd.DataFrame(data=daa2)

fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(range(1,len(df2.Results_Validation)+1), df2['Results_Validation'], color='b')
ax.scatter(range(1,len(df2.Results_Train)+1), df2['Results_Train'], color='r')
plt.grid(False)
plt.show()
