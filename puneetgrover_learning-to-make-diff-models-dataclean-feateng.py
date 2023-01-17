from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torch
from sklearn.metrics import f1_score
def F1score(actuals, preds):
    """
    To get F1 score (macro) for our predictions.
    -----------------------------------------------------------
    Parameters:
        preds: Array of Predicted values
        actuals: Array of Actual labels
    Output:
        Return F1 score (macro)
    """
    return f1_score(actuals, preds, average = 'macro')
PATH = "../input/"

train = pd.read_csv(f'{PATH}train.csv')
test = pd.read_csv(f'{PATH}test.csv')

train.info()
obj_cols = train.columns[train.dtypes == "object"]; obj_cols
train[obj_cols].describe()
# For saving space and compute time (mostly comparison for these)
from sklearn.preprocessing import LabelEncoder
# We will have to use two different label encoders. One for 'Id' and other for 'idhogar'.
lb1 = LabelEncoder()
lb1.fit(list(train['Id'].values))
lb2 = LabelEncoder()
lb2.fit(list(train['idhogar'].values))
# Now we will replace each unique id's with a unique number.
train['Id'] = lb1.transform(list(train['Id'].values))
train['idhogar'] = lb2.transform(list(train['idhogar'].values))

lb3 = LabelEncoder()
lb3.fit(list(test['Id'].values))
lb4 = LabelEncoder()
lb4.fit(list(test['idhogar'].values))
# Now we will replace each unique id's with a unique number.
test['Id'] = lb3.transform(list(test['Id'].values))
test['idhogar'] = lb4.transform(list(test['idhogar'].values))
train['dependency'].unique()  # rate dependency  (yes:1, no:0)
train['dependency'].replace('yes', '1', inplace=True)
train['dependency'].replace('no', '0', inplace=True)
train['dependency'].astype(np.float64);
test['dependency'].replace('yes', '1', inplace=True)
test['dependency'].replace('no', '0', inplace=True)
test['dependency'].astype(np.float64);
train['edjefe'].unique()  # years of education of male head of household  (given, yes:1, no:0)
train['edjefe'].replace('yes', '1', inplace=True)
train['edjefe'].replace('no', '0', inplace=True)
train['edjefe'].astype(np.float64);
test['edjefe'].replace('yes', '1', inplace=True)
test['edjefe'].replace('no', '0', inplace=True)
test['edjefe'].astype(np.float64);
train['edjefa'].unique()  # years of education of female head of household  (given, yes:1, no:0)
train['edjefa'].replace('yes', '1', inplace=True)
train['edjefa'].replace('no', '0', inplace=True)
train['edjefa'].astype(np.float64);
test['edjefa'].replace('yes', '1', inplace=True)
test['edjefa'].replace('no', '0', inplace=True)
test['edjefa'].astype(np.float64);
null_counts = train.isnull().sum()
null_counts[null_counts>0]
test_null_counts = test.isnull().sum()
test_null_counts[test_null_counts>0]
cols = ['Id', 'parentesco1', 'v2a1', 'v18q', 'hacapo', 'rooms', 'r4t3', 'hhsize', 'escolari', 'epared2',
        'epared3', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'Target']
v2a1_null = train.query('v2a1 == "NaN"')[cols]; v2a1_null.shape
# Let us get the family heads of each household in this
v2a1_null_heads = v2a1_null.query('parentesco1 == 1'); v2a1_null_heads.shape
v2a1_null_heads.query('hacapo != 1').shape, v2a1_null_heads.query('hacapo == 1').shape
v2a1_null.query('Target == 1').shape, v2a1_null.query('Target == 2').shape
v2a1_null_heads.query('epared2 != 1 & epared3 != 1').shape # Families who don't have regular or good walls
v2a1_null_heads.query('tipovivi1 != 1').shape # Families who don't own thier own home.
v2a1_null_heads.query('tipovivi2 == 1').shape, v2a1_null_heads.query('tipovivi3 == 1').shape, v2a1_null_heads.query('tipovivi4 == 1').shape, v2a1_null_heads.query('tipovivi5 == 1').shape
v2a1_null_heads.query('tipovivi1 != 1 & tipovivi2 != 1 & tipovivi3 != 1 & tipovivi4 != 1 & tipovivi5 != 1') 
# Checking for any wrong data
v2a1_null_heads.query('(tipovivi4 == 1 | tipovivi5 == 1) & Target == 1').shape, v2a1_null_heads.query('(tipovivi4 == 1 | tipovivi5 == 1) & Target == 2').shape, v2a1_null_heads.query('(tipovivi4 == 1 | tipovivi5 == 1) & Target == 3').shape
train.loc[train['v2a1'].isnull() & train['tipovivi1'] == 1, 'v2a1'] = 0
test.loc[test['v2a1'].isnull() & test['tipovivi1'] == 1, 'v2a1'] = 0
train.query('v2a1 == "NaN"').shape, test.query('v2a1 == "NaN"').shape
train.query('v2a1 != "NaN"')['v2a1'].describe()
a, b = train.query('Target == 1 & v2a1 != "NaN"')['v2a1'].mean(), train.query('Target == 2 & v2a1 != "NaN"')['v2a1'].mean(); a, b
c, d = train.query('Target == 3 & v2a1 != "NaN"')['v2a1'].mean(), train.query('Target == 4 & v2a1 != "NaN"')['v2a1'].mean(); c, d
train.loc[train['v2a1'].isnull() & (train['Target']== 1), 'v2a1'] = a
train.loc[train['v2a1'].isnull() & (train['Target']== 2), 'v2a1'] = b
train.loc[train['v2a1'].isnull() & (train['Target']== 3), 'v2a1'] = c
train.loc[train['v2a1'].isnull() & (train['Target']== 4), 'v2a1'] = d
train.loc[train['v2a1'].isnull()]
test.loc[test['v2a1'].isnull(), 'v2a1'] = (a+b+c+d)/4  # We cannot check Target value here
test.loc[test['v2a1'].isnull()]
v18q1_null = train.query('v18q1 == "NaN"'); v18q1_null.shape
h_ids = v18q1_null['idhogar'].unique(); h_ids.shape
# For every household we will calulate how many of them owns a tablet and put 'v18q1' equal to that sum
for idn in h_ids:
    train.loc[(train['idhogar'] == idn), 'v18q1'] = train.query(f'idhogar == {idn}')['v18q'].sum()
test_v18q1_null = test.query('v18q1 == "NaN"')
h_ids = test_v18q1_null['idhogar'].unique()
for idn in h_ids:
    test.loc[(test['idhogar'] == idn), 'v18q1'] = test.query(f'idhogar == {idn}')['v18q'].sum()
cols = ['Id', 'idhogar', 'escolari', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4',
        'estadocivil5', 'estadocivil6', 'estadocivil7', 'instlevel1', 'age', 'Target']
cols2 = cols[:-1]
rez_esc_null = train.query('rez_esc == "NaN"')[cols]
test_rez_esc_null = test.query('rez_esc == "NaN"')[cols2]; rez_esc_null.shape, test_rez_esc_null.shape
rez_esc_null.query('instlevel1 == 1').shape, test_rez_esc_null.query('instlevel1 == 1').shape
train.loc[(train['rez_esc'].isnull()) & (train['instlevel1'] == 1), 'rez_esc'] = 0
test.loc[(test['rez_esc'].isnull()) & (test['instlevel1'] == 1), 'rez_esc'] = 0
rez_esc_null = train.query('rez_esc == "NaN"')[cols]
test_rez_esc_null = test.query('rez_esc == "NaN"')[cols2]; rez_esc_null.shape, test_rez_esc_null.shape
# estadocivil1: =1 if less than 10 years old
rez_esc_null.query('estadocivil1 == 1').shape, test_rez_esc_null.query('estadocivil1 == 1').shape
rez_esc_null.query('estadocivil2 == 1').shape, rez_esc_null.query('estadocivil3 == 1').shape, rez_esc_null.query('estadocivil4 == 1').shape
test_rez_esc_null.query('estadocivil2 == 1').shape, test_rez_esc_null.query('estadocivil3 == 1').shape, test_rez_esc_null.query('estadocivil4 == 1').shape
rez_esc_null.query('estadocivil5 == 1').shape, rez_esc_null.query('estadocivil6 == 1').shape, rez_esc_null.query('estadocivil7 == 1').shape
test_rez_esc_null.query('estadocivil5 == 1').shape, test_rez_esc_null.query('estadocivil6 == 1').shape, test_rez_esc_null.query('estadocivil7 == 1').shape
a = train.loc[(~train['rez_esc'].isnull()) & (train['estadocivil2'] == 1) & (train['escolari'] > 0)]['rez_esc'].mean()
b = train.loc[(~train['rez_esc'].isnull()) & (train['estadocivil3'] == 1) & (train['escolari'] > 0)]['rez_esc'].mean()
c = train.loc[(~train['rez_esc'].isnull()) & (train['estadocivil4'] == 1) & (train['escolari'] > 0)]['rez_esc'].mean()
d = train.loc[(~train['rez_esc'].isnull()) & (train['estadocivil5'] == 1) & (train['escolari'] > 0)]['rez_esc'].mean()
e = train.loc[(~train['rez_esc'].isnull()) & (train['estadocivil6'] == 1) & (train['escolari'] > 0)]['rez_esc'].mean()
f = train.loc[(~train['rez_esc'].isnull()) & (train['estadocivil7'] == 1) & (train['escolari'] > 0)]['rez_esc'].mean()
a, b, c, d, e, f

train.loc[(train['rez_esc'].isnull()) & (train['estadocivil2'] == 1), 'rez_esc'] = 3
train.loc[(train['rez_esc'].isnull()) & (train['estadocivil3'] == 1), 'rez_esc'] = 0
train.loc[(train['rez_esc'].isnull()) & (train['estadocivil4'] == 1), 'rez_esc'] = 0
train.loc[(train['rez_esc'].isnull()) & (train['estadocivil5'] == 1), 'rez_esc'] = 2
train.loc[(train['rez_esc'].isnull()) & (train['estadocivil6'] == 1), 'rez_esc'] = 0
train.loc[(train['rez_esc'].isnull()) & (train['estadocivil7'] == 1), 'rez_esc'] = 1

test.loc[(test['rez_esc'].isnull()) & (test['estadocivil2'] == 1), 'rez_esc'] = 3
test.loc[(test['rez_esc'].isnull()) & (test['estadocivil3'] == 1), 'rez_esc'] = 0
test.loc[(test['rez_esc'].isnull()) & (test['estadocivil4'] == 1), 'rez_esc'] = 0
test.loc[(test['rez_esc'].isnull()) & (test['estadocivil5'] == 1), 'rez_esc'] = 2
test.loc[(test['rez_esc'].isnull()) & (test['estadocivil6'] == 1), 'rez_esc'] = 0
test.loc[(test['rez_esc'].isnull()) & (test['estadocivil7'] == 1), 'rez_esc'] = 1
train['rez_esc'].isnull().sum(), test['rez_esc'].isnull().sum()
cols = ['Id', 'idhogar', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5',
       'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']
# We will just put this value equal to avg year of education of adults with the help of selected cols
meaneduc_null = train.query('meaneduc == "NaN"')[cols]
test_meaneduc_null = test.query('meaneduc == "NaN"')[cols]
h_ids = meaneduc_null['idhogar'].unique(); h_ids
print(train.loc[(train['idhogar'] ==  326), 'meaneduc'].values)
print(train.loc[(train['idhogar'] == 1959), 'meaneduc'].values) 
print(train.loc[(train['idhogar'] == 2908), 'meaneduc'].values)
def meaneduc_correction(null_view, df, hids):
    """
    Function to correct null_values in "meaneduc" feature. Will put them equal to mean, after calculating it
    using "instlevel"'s.
    ---------------------------------------------------------------------------------------------------------
    Parameters:
        null_view: View of origianl dataframe with null values of "meaneduc"
        df: Original DataFrame
        hids: Unique Household ids of households with null "meaneduc"
    """
    for idn in hids:
        # Number of people with no education and so on
        a = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel1'] == 1)].shape[0] # No ed
        b = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel2'] == 1)].shape[0] # Inc. prim
        c = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel3'] == 1)].shape[0] # Com. prim
        d = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel4'] == 1)].shape[0] # Inc Acad Sec L.
        e = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel5'] == 1)].shape[0] # Com Acad Sec L.
        f = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel6'] == 1)].shape[0] # Inc Tech Sec L.
        g = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel7'] == 1)].shape[0] # Com Tech Sec L.
        h = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel8'] == 1)].shape[0] # UndGrad n HigEd
        i = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel9'] == 1)].shape[0] # Postgrad

        mean_educ = (a*0 + b*4 + c*8 + d*2 + e*4 + f + g*2 + h*4 + i) / (a+b+c+d+e+f+g+h+i)

        df.loc[(df['meaneduc'].isnull()) & (df['idhogar'] == idn), 'meaneduc'] =  mean_educ
        df.loc[(df['SQBmeaned'].isnull()) & (df['idhogar'] == idn), 'SQBmeaned'] =  mean_educ**2
meaneduc_correction(meaneduc_null, train, h_ids)
null_counts = train.isnull().sum()
null_counts[null_counts>0]
h_ids = test_meaneduc_null['idhogar'].unique(); h_ids
meaneduc_correction(test_meaneduc_null, test, h_ids)
test_null_counts = test.isnull().sum()
test_null_counts[test_null_counts>0]
train.shape, test.shape
train['Id'].unique().size, test['Id'].unique().size  # So, Ids are unique
# Now for the second part
cols = ['v2a1', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q1',
       'r4h3', 'r4m3', 'r4t3', 'tamhog', 'tamviv', 'hhsize', 'paredblolad',
       'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc',
       'paredfibras', 'paredother', 'pisomoscer', 'pisocemento', 
       'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera', 'techozinc',
       'techoentrepiso', 'techocane', 'techootro', 'cielorazo',
       'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 
       'planpri', 'noelec', 'coopele', 'sanitario1', 'sanitario2', 
       'sanitario3', 'sanitario5', 'sanitario6', 
       'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4',
       'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5',
       'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 
       'etecho3', 'eviv1', 'eviv2', 'eviv3', 'hogar_nin', 'hogar_adul',
       'hogar_mayor', 'hogar_total', 'dependency', 'meaneduc', 'bedrooms', 
       'overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4',
       'tipovivi5', 'computer', 'television', 'qmobilephone', 'lugar1',
       'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'Target']
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

def check_for_wrong_data(data, columns, labelE, gpby='idhogar'):
    """
    Checks for data mismatches in rows of every group, which we get by gpby (groupby)
    feature, on columns in "columns". If mismatch is there, put it equal to value in
    columns of head of the household and print a message for this.
    ----------------------------------------------------------------------------------
    Input:
        data : Train or Test set or their sliced views
        columns : columns to check for corrupted data
        labelE : Label encoder of "data"'s  "idhogar" column used in inverse_transform 
        gpby : feature to group by to check for diff "cols" in that group
    Output:
        Return four arrays:
        1) Array with ids of households with no head
        2) Array with ids of households with wrong data
        3) Array of arrays with column name with wrong data for each household in (2)nd array
        4) Array of arrays with ids of members with wrong data for each household in (2)nd array
    """
    id_head_zero = [] # Will contain house ids with no head
    idhogarId_f = []
    cols_f = []
    mem_f = []
    grouped = data.groupby(gpby, sort=True)
    for gid in range(len(grouped)):
        members = grouped.get_group(gid)
        h_Head = members.loc[(members['parentesco1'] == 1)]
        if h_Head.shape[0] == 0:
            id_head_zero.append(members['idhogar'].values[0])
            continue
        idhogarId_w = []
        cols_t = []
        mem_t = []
        if members.shape[0] > 1:
            for col in columns:
                for m in members.iterrows():
                    if h_Head[col].values[0] != m[1][col]:
                        if h_Head['idhogar'].values[0] not in idhogarId_w : idhogarId_w.append(h_Head['idhogar'].values[0])
                        if col not in cols_t : cols_t.append(col)
                        if m[1]['Id'] not in mem_t : mem_t.append(m[1]['Id'])
                        # Correct this column
                        data.loc[(train['Id'] == m[1]['Id']), col] = h_Head[col].values[0]
        idhogarId_f.append(idhogarId_w); cols_f.append(cols_t); mem_f.append(mem_t)
        if len(idhogarId_w) > 0:
            for i in range(len(idhogarId_w)):
                print("Household with Id: "
                +str(labelE.inverse_transform([idhogarId_w[i]])[0])
                +" has " + str(len(mem_t)) + " member(s) with diff. value(s) of " + str(len(cols_t)) + " column(s)."
                + " " + str(cols_t) )
    return id_head_zero, idhogarId_f, cols_f, mem_f
id_head_zero, *_ = check_for_wrong_data(train, cols, lb2)
train.loc[(train['idhogar'] == id_head_zero[11])]   # 4, 6, 7, 8, 11 have more than 1 persons in home but no head
cols = cols[:-1] # Remove "Target"
id_head_zero, *_ = check_for_wrong_data(test, cols, lb4)
import seaborn as sns
columns = train.select_dtypes('number').drop(['Id', 'idhogar', 'Target'], axis=1).columns

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(20, 15))
fig.subplots_adjust(top=1.3)
#train.loc[:,columns[1:22]].boxplot(ax=axes[0])
a = sns.boxplot(x = "variable", y = "value", data = pd.melt(train.loc[:,columns[1:22]]), ax=axes[0])
b = sns.boxplot(x = "variable", y = "value", data = pd.melt(train.loc[:,columns[22:70]]), ax=axes[1])
b.set_xticklabels(rotation=30, labels = columns[22:70]);
c = sns.boxplot(x = "variable", y = "value", data = pd.melt(train.loc[:,columns[70:98]]), ax=axes[2])
c.set_xticklabels(rotation=30, labels = columns[70:120]);
d = sns.boxplot(x = "variable", y = "value", data = pd.melt(train.loc[:,columns[99:120]]), ax=axes[3])
possible_outliers = [columns[0]] + [columns[98]]; columns[0], columns[98]
sns.boxplot(data = train[possible_outliers[0]])
train.loc[(train['v2a1'] > 300000), ['idhogar', 'v2a1', 'Target']].query("Target != 4")  # Actually, all above 300,000 are from "Target" of 4
train.loc[(train['v2a1'] > 2000000), ['idhogar', 'v2a1', 'Target']]   # So leave it
sns.boxplot(data = train[possible_outliers[1]])
train.loc[(train['meaneduc'] > 25), ['idhogar', 'Target', 'meaneduc']].query("Target != 4")
columns = ['v2a1', 'rooms', 'tamhog', 'overcrowding', 'v18q1', 'r4t3', 'meaneduc', 'qmobilephone', 'Target']
sns.pairplot(train[columns])
train.drop(['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding',
            'SQBdependency', 'SQBmeaned', 'agesq'], axis = 1, inplace=True)
test.drop(['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding',
            'SQBdependency', 'SQBmeaned', 'agesq'], axis = 1, inplace=True)
# Plotting a heat map
import seaborn as sns
plt.subplots(figsize=(20,15))
sns.heatmap(train.corr().abs(), cmap="BuPu")
DropCols = ['energcocinar1', 'energcocinar4', 'energcocinar3', 'energcocinar2', 'epared1', 'epared2', 'epared3',
        'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', #'instlevel1', 'instlevel2', 'instlevel3', 
        #'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9',
        'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']

train['CookingType'] = np.argmax(np.array(train[[ 'energcocinar1', 'energcocinar4', 'energcocinar3', 'energcocinar2' ]]), axis=1)
train['WallType'] = np.argmax(np.array(train[[ 'epared1', 'epared2', 'epared3' ]]), axis=1)
train['RoofType'] = np.argmax(np.array(train[[ 'etecho1', 'etecho2', 'etecho3' ]]), axis=1)
train['FloorType'] = np.argmax(np.array(train[[ 'eviv1', 'eviv2', 'eviv3' ]]), axis=1)
# EdLevel is being removed during deletion of highly correlated features
# train['EdLevel'] = np.argmax(np.array(train[ [ 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9' ]]), axis=1)
train['HouseType'] = np.argmax(np.array(train[[ 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5' ]]), axis=1)

test['CookingType'] = np.argmax(np.array(test[[ 'energcocinar1', 'energcocinar4', 'energcocinar3', 'energcocinar2' ]]), axis=1)
test['WallType'] = np.argmax(np.array(test[[ 'epared1', 'epared2', 'epared3' ]]), axis=1)
test['RoofType'] = np.argmax(np.array(test[[ 'etecho1', 'etecho2', 'etecho3' ]]), axis=1)
test['FloorType'] = np.argmax(np.array(test[[ 'eviv1', 'eviv2', 'eviv3' ]]), axis=1)
# EdLevel is being removed during deletion of highly correlated features
# test['EdLevel'] = np.argmax(np.array(test[[ 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9' ]]), axis=1)
test['HouseType'] = np.argmax(np.array(test[[ 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5' ]]), axis=1)

train.drop(DropCols, axis=1, inplace=True)
test.drop(DropCols, axis=1, inplace=True)
test.shape, train.shape
# Per member features
train['phones-per-mem'] = train['qmobilephone'] / train['tamviv']
train['tablets-per-mem'] = train['v18q1'] / train['tamviv']
train['rooms-per-mem'] = train['rooms'] / train['tamviv']
train['rent-per-adult'] = train['v2a1'] / train['hogar_adul']

test['phones-per-mem'] = test['qmobilephone'] / test['tamviv']
test['tablets-per-mem'] = test['v18q1'] / test['tamviv']
test['rooms-per-mem'] = test['rooms'] / test['tamviv']
test['rent-per-adult'] = test['v2a1'] / test['hogar_adul']
def chk_n_remove_corr(df):
    """
    Checks for highly correlated features and removes them.
    ---------------------------------------------------------------------
    Parameters:
        df: Dataframe to check for correlation
    Output:
        Return list of removed features/columns.
    """
    corr_matrix = train.corr()
    
    # Taking only the upper triangular part of correlation matrix: (We want to remove only one of corr features)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.975
    to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.975)]
    
    train.drop(to_drop, axis=1, inplace=True)
    return to_drop
to_drop = chk_n_remove_corr(train)
to_drop
test.drop(to_drop, axis=1, inplace=True)
train.shape, test.shape
hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6',
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1']

hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1','tamviv','hogar_nin',# 'hhsize', 'tamhog',
              'CookingType', 'WallType', 'RoofType', 'HouseType' , 'FloorType', 
              'hogar_adul','hogar_mayor',  'bedrooms', 'qmobilephone'] # ,'hogar_total']

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding',
          'phones-per-mem', 'tablets-per-mem', 'rooms-per-mem', 'rent-per-adult']

ind_bool = ['v18q', 'dis', 'male', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5',
            'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9',
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'mobilephone']

ind_ordered = ['age', 'escolari', 'rez_esc']#, 'EdLevel']
train[hh_bool + ind_bool] = train[hh_bool + ind_bool].astype(bool)
test[hh_bool + ind_bool] = test[hh_bool + ind_bool].astype(bool)
train[hh_cont] = train[hh_cont].astype('float64')
test[hh_cont] = test[hh_cont].astype('float64');
train[hh_ordered + ind_ordered] = train[hh_ordered + ind_ordered].astype(int)
test[hh_ordered + ind_ordered] = test[hh_ordered + ind_ordered].astype(int);
train['Target'] = train['Target'].astype(int);
train.isnull().sum()[train.isnull().sum() > 0], test.isnull().sum()[test.isnull().sum() > 0]
# Only one value in train and 10 in test
train.loc[train['rent-per-adult'].isnull(), 'rent-per-adult'] = train.loc[train['rent-per-adult'].isnull(), 'v2a1'].values[0]
test.loc[test['rent-per-adult'].isnull(), 'rent-per-adult'] = test.loc[test['rent-per-adult'].isnull(), 'v2a1'].values[0]
train.isnull().sum()[train.isnull().sum() > 0], test.isnull().sum()[test.isnull().sum() > 0]
for c in train.columns:
    if train[c].dtype != 'float64': continue
    s = np.where(train[c].values >= np.finfo(np.float32).max)
    if len(s[0])>0:
        print(c)
        print(s)
train[train['rent-per-adult'] > np.finfo(np.float32).max][['rent-per-adult']]
train[train['rent-per-adult'] > np.finfo(np.float32).max][['Id', 'idhogar', 'v2a1', 'hogar_adul', 'age']]
train[train['idhogar']==1959][['idhogar', 'v2a1', 'age']]
train[train['idhogar']==2908][['idhogar', 'v2a1', 'age']]
h_ids = train[train['rent-per-adult'] > np.finfo(np.float32).max]['idhogar'].unique()
for h_id in h_ids:
    rent_per_adul = train.loc[(train['idhogar']==h_id), 'v2a1'].values[0] / train.loc[(train['idhogar']==h_id)].shape[0]
    # Assuming the rent is being divided among them equally
    train.loc[train['idhogar']==h_id, 'rent-per-adult'] = rent_per_adul
    train.loc[train['idhogar']==h_id, 'rent-per-adult_sum'] = train.loc[(train['idhogar']==h_id), 'v2a1'].values[0]
for c in test.columns:
    if test[c].dtype != 'float64': continue
    s = np.where(test[c].values >= np.finfo(np.float32).max)
    if len(s[0])>0:
        print(c)
        print(s)
test[test['rent-per-adult'] > np.finfo(np.float32).max][['rent-per-adult']]
test[test['rent-per-adult'] > np.finfo(np.float32).max][['Id', 'idhogar', 'v2a1', 'hogar_adul', 'age']]
h_ids = test[test['rent-per-adult'] > np.finfo(np.float32).max]['idhogar'].unique()

for h_id in h_ids:
    rent_per_adul = test.loc[(test['idhogar']==h_id), 'v2a1'].values[0] / test.loc[(test['idhogar']==h_id)].shape[0]
    # Assuming the rent is being divided among them equally
    test.loc[test['idhogar']==h_id, 'rent-per-adult'] = rent_per_adul
    test.loc[test['idhogar']==h_id, 'rent-per-adult_sum'] = test.loc[(test['idhogar']==h_id), 'v2a1'].values[0]
def make_new_features_grouping(df, dtypes, gpby, customAggFunc=None):
    """
    Make new features aggregating on groups found by "gbpy".
    -----------------------------------------------------------------
    Parameters:
        df: Dataset for which new features are to be made
        dtypes: Data Types of features which will be used to create new features (string, type or array)
                eg: bool, 'number', 'float' etc
        gbpy: Feature on which grouping will be done
        customAggFunc: A custom Aggregation function or a list of such functions
    Output: 
        Returns Original DataFrame with new features
    """
    # Grouping
    if 'Target' in df.columns: numeric_type = df.select_dtypes(dtypes).drop(['Target', 'Id'], axis=1).copy()
    else: numeric_type = df.select_dtypes(dtypes).drop(['Id'], axis=1).copy()
    
    funcs = ['count', 'mean', 'max', 'min', 'sum', 'std', 'var', 'quantile']
    
    if customAggFunc is None: new = numeric_type.groupby(gpby).agg(funcs)
    elif isinstance(customAggFunc, list): new = numeric_type.groupby(gpby).agg(funcs + customAggFunc)
    else: new = numeric_type.groupby(gpby).agg(funcs + [customAggFunc])
    
    # Rename all columns and remove levels
    columns = []
    for old_col in new.columns.levels[0]:
        if old_col != 'idhogar':
            for new_col in new.columns.levels[1]:
                columns.append(old_col + '_' + new_col)
    new.columns = columns
    
    return df.merge(new.reset_index(), on="idhogar", how='left')
train.shape, test.shape
%time train = make_new_features_grouping(train, ['number', bool], "idhogar")
%time test = make_new_features_grouping(test, ["number", bool], "idhogar")
train.shape, test.shape
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
to_drop = chk_n_remove_corr(train)
len(to_drop), 'Target' in to_drop
test.drop(to_drop, axis=1, inplace=True)
train.shape, test.shape
from sklearn.preprocessing import RobustScaler
scaler1 = RobustScaler()
scaler2 = RobustScaler()

scaled1 = scaler1.fit_transform(train.drop(['Target', 'Id', 'idhogar'], axis=1))
scaled2 = scaler2.fit_transform(test.drop(['Id', 'idhogar'], axis=1))

cols1 = train.drop(['Target', 'Id', 'idhogar'], axis=1).columns
cols2 = test.drop(['Id', 'idhogar'], axis=1).columns

trPCA = pd.DataFrame(scaled1, index=np.arange(train.shape[0]), columns=cols1)
tsPCA = pd.DataFrame(scaled2, index=np.arange(test.shape[0]), columns=cols2)
from sklearn.decomposition import PCA
pca = PCA(n_components=5, svd_solver='full')
transformed1 = pca.fit_transform(trPCA)
transformed2 = pca.transform(tsPCA)
for i in range(5):
    train[f'PCA{i+1}'] = transformed1[:,i]
    test[f'PCA{i+1}'] = transformed2[:,i]
train.shape, test.shape
pca.explained_variance_ratio_
train['Target'].value_counts().plot.barh()
train['Target'].value_counts(), 774+1221+1558+1500
rows1 = (train['Target'] == 1)
rows2 = (train['Target'] == 2)
rows3 = (train['Target'] == 3)
rows123 = (rows1 | rows2 | rows3)
rows4 = (train['Target'] == 4)
rows123.sum(), rows4.sum()
# We will take only first count1+count2 rows. Where count1 will go to train and count2 will go to validation set.
def train_val_split(rows, tvlen=None, vper=None):
    """
    Takes in "row" array which is location matrix for specific category(say) and 
    divides it into "train row" and "validation row" of locations. If you only want
    limited rows from "rows" then specify tvlen, which is a tuple of number of rows
    you want in train and validaion set.
    -----------------------------------------------------------------------------------
    Parameters:
        rows = An array of specific selected rows. (Where ith row is true if selected)
        tvlen = An array or a tuple of number of elements in train and val. set
        vper = perecent of elements you want in Validation set (Use it if you want all rows 
                to be divided into test and val sets from the "rows" Array or pd.Series)
    Output:
        Returns two Arrays or pd.Series of selected rows for train and validation set
        where ith element is True if that row is selected.
    """
    if tvlen is not None and vper is None:
        count1 = tvlen[0]
        count2 = tvlen[1]
    elif tvlen is None and vper is not None:
        c = rows.sum()
        count1 = int((1-vper)*c)
        count2 = int(vper*c)
    else:
        raise Exception('One of "tvlen" or "vper" should be given.')
    
    rowst, rowsv = rows.copy(), rows.copy()
    
    for i in range(len(rows)):
        
        # If we have taken count1 rows in training set, put all values equal to False. (after, count1 == 0)
        if not count1:
            rowst[i] = False
            # If we have got count2 rows in validation set, set all others equal to False.
            if not count2:
                rowsv[i] = False
            # Don't do anything to fisrt count2 rows after first count1 rows of training set,
            # where Target = selected Target and dec. count2
            count2 -= rowsv[i] # As True = 1 and False = 0
            continue
        # Equal to False because they will be in Training set
        rowsv[i] = False
        # Don't do anything to fisrt count2 rows, where Target = selected Target, and dec. count1
        count1 -= rowst[i]
    
    return rowst, rowsv
rows123t, rows123v = train_val_split(rows123, vper=0.1)
rows4t, rows4v = train_val_split(rows4, tvlen=(1300, 200))
rows123t.sum(), rows123v.sum(), rows4t.sum(), rows4v.sum()
train.drop(['Id', 'idhogar'], axis=1, inplace=True)
xtrain, xvalid = train.loc[rows123t|rows4t].drop('Target', axis=1).copy(), train.loc[rows123v|rows4v].drop('Target', axis=1).copy()
ytrain, yvalid = train['Target'].loc[rows123t|rows4t].copy(), train['Target'].loc[rows123v|rows4v].copy()
xtrain.shape, ytrain.shape, xvalid.shape, yvalid.shape
xtrain.head()
ytrain.value_counts()
yvalid.value_counts()
ytrain.value_counts().plot.barh()
train['Target'].value_counts()
target1 = train.loc[train['Target']==1].copy()
target2 = train.loc[train['Target']==2].copy()
target3 = train.loc[train['Target']==3].copy()
target1 = pd.concat([target1]*8, ignore_index=True).copy(); target1.shape
target2 = pd.concat([target2]*4, ignore_index=True).copy(); target2.shape
target3 = pd.concat([target3]*5, ignore_index=True).copy(); target3.shape
train2 = train.copy()
train2 = pd.concat([train2, target1, target2, target3], ignore_index=True); train2.shape
train2 = train2.sample(frac=1).reset_index(drop=True)
xtrain2, xvalid2 = train2.iloc[:15000].drop(['Target'], axis=1).copy(), train2.iloc[15000:].drop(['Target'], axis=1).copy()
ytrain2, yvalid2 = train2.iloc[:15000]['Target'].copy(), train2.iloc[15000:]['Target'].copy()
xtrain2.shape, ytrain2.shape, xvalid2.shape, yvalid2.shape
train2['Target'].value_counts().plot.barh()
from sklearn.ensemble import RandomForestClassifier
import math
def print_score(m, trn, val):
    """
    Print F1 score for training set and validation set, where m is a
    RandomForestClassifier.
    ----------------------------------------------------------------------
    Parameters:
        m: RandomForestClassifier model
        trn: tuple or array of Input and Output training data points
        val: tuple or array of Input and Output validation data points
    """
    print("Train F1score: ", str(F1score(trn[1], m.predict(trn[0]))),
    ",  Valid. F1score: ", str(F1score(val[1], m.predict(val[0]))))
    print("Train Acc.: ", str(m.score(trn[0], trn[1])),
    ", Valid. Acc.: ", str(m.score(val[0], val[1])))
m = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.7, n_jobs=-1)
m.fit(xtrain, ytrain)
print_score(m, (xtrain, ytrain), (xvalid, yvalid))
# Here I increased min_sample_leaf hyperparameter
m2 = RandomForestClassifier(n_estimators=100, min_samples_leaf=150, max_features=0.5, n_jobs=-1)
m2.fit(xtrain2, ytrain2)
print_score(m2, (xtrain2, ytrain2), (xvalid2, yvalid2))
from sklearn.model_selection import train_test_split
a, b, c, d = train_test_split(train.drop('Target', axis=1), train['Target'], test_size=0.20,
                                                    stratify=train['Target'])
xtrain3, xvalid3, ytrain3, yvalid3 = a.copy(), b.copy(), c.copy(), d.copy()
m3 = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1, class_weight='balanced')
m3.fit(xtrain3, ytrain3)
print_score(m3, (xtrain3, ytrain3), (xvalid3, yvalid3))
def plot_bar_stacked(y, preds):
    """
    For plotting predictions, right and wrong. For wrong predictions it will
    plot stacked bars in diff. colors denoting the class to which it was 
    misplaced.
    -------------------------------------------------------------------------
    Parameters:
        y : actual ouput values
        preds : predicted output values
    Output:
        Plot a stacked graph of count of right and wrong
    """
    # Output Categories 
    categories = np.array([1, 2, 3, 4])
    # This will keep count of right predictions and count of wrong prediction in each category
    counts = [[0, 0, 0, 0], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    # Calculating wrong and right predictions for all categories
    for cat in categories:
        index = (y == cat)
        right = (preds[index] == y[index]).sum()
        # For wrong preds:
        p = preds[index]
        w1 = (p[(p != y[index])] == 1).sum()
        w2 = (p[(p != y[index])] == 2).sum()
        w3 = (p[(p != y[index])] == 3).sum()
        w4 = (p[(p != y[index])] == 4).sum()

        counts[1][cat-1] = [w1, w2, w3, w4]
        counts[0][cat-1] = right
        
    # Plotting
    ind = np.arange(4)
    width = 0.15

    fig, ax = plt.subplots(figsize=(15,10), sharey=True)
    
    # Quite a simple way to plot stacked bar plot
    df = pd.DataFrame(counts[1], index=np.arange(1, 5), columns=['W Pred=1', 'W Pred=2', 'W Pred=3', 'W Pred=4'])
    df.plot.bar(ax=ax, width=width, stacked=True, colormap='RdYlBu')

    ax.bar(ind+width, counts[0], width=-width, color='green', label='Right')

    ax.set(xticks=ind + width, xticklabels=categories, xlim=[2*width - 1, 4])
    ax.legend()
preds = m.predict(xvalid)
plot_bar_stacked(yvalid, preds)
# It is not necessary that it will generalize well for test set too. (Though this is 
# giving me better results on public leaderboard.)
preds = m2.predict(xvalid2)
plot_bar_stacked(yvalid2, preds)
preds = m3.predict(xvalid3)
plot_bar_stacked(yvalid3, preds)
fi = pd.DataFrame({'cols':xtrain.columns, 'imp':m.feature_importances_}).sort_values('imp', ascending=False); fi[0:10]
fi.plot('cols', 'imp', figsize=(10, 6), legend=False)
to_keep = fi.loc[(fi['imp']>0.005), 'cols']; to_keep.shape
xtrain_new, xvalid_new = xtrain[to_keep].copy(), xvalid[to_keep].copy()
m = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features=0.5, n_jobs=-1)
m.fit(xtrain_new, ytrain)
print_score(m, (xtrain_new, ytrain), (xvalid_new, yvalid))
fi = pd.DataFrame({'cols':xtrain2.columns, 'imp':m2.feature_importances_}).sort_values('imp', ascending=False); fi[0:10]
fi.plot('cols', 'imp', figsize=(10, 6), legend=False)
to_keep = fi.loc[(fi['imp']>0.005), 'cols']; to_keep.shape
xtrain2_new, xvalid2_new = xtrain2[to_keep].copy(), xvalid2[to_keep].copy()
m2 = RandomForestClassifier(n_estimators=100, min_samples_leaf=150, max_features=0.5, n_jobs=-1)
m2.fit(xtrain2_new, ytrain2)
print_score(m2, (xtrain2_new, ytrain2), (xvalid2_new, yvalid2))
fi = pd.DataFrame({'cols':xtrain3.columns, 'imp':m3.feature_importances_}).sort_values('imp', ascending=False); fi[0:10]
to_keep = fi.loc[(fi['imp']>0.005), 'cols']; to_keep.shape
xtrain3_new, xvalid3_new = xtrain3[to_keep].copy(), xvalid3[to_keep].copy()
m3 = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1, class_weight="balanced")
m3.fit(xtrain3_new, ytrain3)
print_score(m3, (xtrain3_new, ytrain3), (xvalid3_new, yvalid3))
import scipy
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(xtrain_new).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,15))
dendrogram = hc.dendrogram(z, labels=xtrain_new.columns, orientation='left', leaf_font_size=16)
plt.show()
m = RandomForestClassifier(n_estimators=50, min_samples_leaf=25, max_features=0.6, n_jobs=-1, oob_score=True)
m.fit(xtrain_new, ytrain)
print(m.oob_score_)
check_with = m.oob_score_
#scores = []
#for col in ['v2a1_sum', 'v2a1']:
#    m = RandomForestClassifier(n_estimators=50, min_samples_leaf=25, max_features=0.6, n_jobs=-1, oob_score=True)
#    m.fit(xtrain_new.drop(col, axis=1), ytrain)
#    scores.append(m.oob_score_)
#    print(m.oob_score_)
#to_drop = []
#for i, col in enumerate(['v2a1_sum', 'v2a1']):
#    if scores[i] > check_with: to_drop.append(col)
#xtrain_new, xvalid_new = xtrain_new.drop(to_drop, axis=1), xvalid_new.drop(to_drop, axis=1) 
m = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1)
m.fit(xtrain_new, ytrain)
print_score(m, (xtrain_new, ytrain), (xvalid_new, yvalid))
preds = m.predict(xvalid_new)
plot_bar_stacked(yvalid, preds)
pass
# from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
# It needs labels from 0 to n-1, where n is number of classes
#ytrain3 = ytrain3-1
#yvalid3 = yvalid3-1
# For decreasing learning rate of model with time
def learningRateAnnl(current_iter):
    base_learning_rate = 0.1
    min_learning_rate = 0.02
    lr = base_learning_rate  * np.power(.995, current_iter)
    return max(lr, min_learning_rate)

def evaluate_macroF1_lgb(truth, predictions):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', f1, True)
fit_params={"early_stopping_rounds":300, 
            "eval_metric" : evaluate_macroF1_lgb, 
            "eval_set" : [(xvalid3, yvalid3.copy()-1)],
            'eval_names': ['valid'],
            'callbacks': [lgb.reset_parameter(learning_rate=learningRateAnnl)],
            'verbose': False,
            'categorical_feature': 'auto'}
from scipy.stats import randint
from scipy.stats import uniform
param_test ={'num_leaves': randint(12, 20), 
             'min_child_samples': randint(40, 120), 
             #'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': uniform(loc=0.75, scale=0.20), 
             'colsample_bytree': uniform(loc=0.8, scale=0.15),
             #'reg_alpha': [0, 1e-3, 1e-1, 1, 10, 50, 100],
             #'reg_lambda': [0, 1e-3, 1e-1, 1, 10, 50, 100],
             #'boosting': ['dart', 'goss', 'gbdt']
            }
maxHPs = 400
classifier = lgb.LGBMClassifier(learning_rate=0.05, n_jobs=-1, n_estimators=500, objective='multiclass')

#rs = RandomizedSearchCV(estimator= classifier, param_distributions=param_test, n_iter=maxHPs,
#                        scoring='f1_macro', cv=5, refit=True, verbose=True)
#_ = rs.fit(xtrain3, (ytrain3).copy()-1, **fit_params)
#opt_parameters = rs.best_params_; opt_parameters
# op_parameters found by above method (Random Search)
opt_parameters = {'colsample_bytree': 0.8755593602517565,
 'min_child_samples': 51,
 'num_leaves': 19,
 'subsample': 0.9437154452377117}
classifier = lgb.LGBMClassifier(**classifier.get_params())
classifier.set_params(**opt_parameters)

fit_params['verbose'] = 200
_ = classifier.fit(xtrain3, (ytrain3).copy() -1, **fit_params)
kfold = 5
kf = StratifiedKFold(n_splits=kfold, shuffle=True)

for trn_idx, tst_idx in kf.split(train.drop(['Target'], axis=1), train['Target']):
    xtr, xval = train.drop(['Target'], axis=1).iloc[trn_idx], train.drop(['Target'], axis=1).iloc[tst_idx]
    ytr, yval = train['Target'].iloc[trn_idx].copy() -1, train['Target'].iloc[tst_idx].copy() -1
    
    classifier.fit(xtr, ytr, eval_set=[(xval, yval)], 
            early_stopping_rounds=300, verbose=200)
preds = classifier.predict(xvalid)
((preds+1) == yvalid).sum()/len(preds)
plot_bar_stacked(yvalid, preds+1)
classifier2 = lgb.LGBMClassifier(**classifier.get_params())

kfold = 5
kf = StratifiedKFold(n_splits=kfold, shuffle=True)

for trn_idx, tst_idx in kf.split(train2.drop(['Target'], axis=1), train2['Target']):
    xtr, xval = train2.drop(['Target'], axis=1).iloc[trn_idx].copy(), train2.drop(['Target'], axis=1).iloc[tst_idx].copy()
    ytr, yval = train2['Target'].iloc[trn_idx].copy()-1, train2['Target'].iloc[tst_idx].copy() -1
    
    classifier2.fit(xtr, ytr, eval_set=[(xval, yval)], 
            early_stopping_rounds=300, verbose=200)
# Won't necessarily generalize well.
preds = classifier2.predict(xvalid2)
plot_bar_stacked(yvalid2, preds+1)
pass
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd.variable import Variable
Ttrain = TensorDataset(torch.DoubleTensor(np.array(xtrain.values, dtype="float32")), #.cuda
                       torch.LongTensor(np.array(ytrain.values, dtype="float32")-1)) #.cuda
trainLoader = DataLoader(Ttrain, batch_size = 20, shuffle=True)
class Net(nn.Module):
    def __init__(self, n_cols):
        super(Net, self).__init__()
        
        self.first = nn.Sequential(
            nn.BatchNorm1d(n_cols),
            nn.Linear(n_cols, 10),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            #nn.Linear(80, 80),
            #nn.BatchNorm1d(80),
            #nn.ReLU(),
            #nn.Linear(80, 80),
            #nn.BatchNorm1d(80),
            #nn.ReLU(),
            #nn.Dropout(p=0.25),
            #nn.Linear(80, 20),
            #nn.BatchNorm1d(20),
            #nn.ReLU(),
            #nn.Linear(50, 20),
            #nn.ReLU(),
            nn.Linear(10, 4),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.first(x)
net = Net(len(xtrain.columns)).double() #.cuda()
loss = nn.CrossEntropyLoss()
metrics = [F1score]
#opt = optim.SGD(net.parameters(), 1e-3, momentum=0.999, weight_decay=1e-3, nesterov=True)
opt = optim.Adam(net.parameters(), weight_decay=1e-3)
#opt = optim.RMSprop(net.parameters(), momentum=0.9, weight_decay=1e-3)
def fit(model, lr, xtr, ytr, xvl, yvl, train_dl, n_epochs, loss, opt, metrics, annln=False, mult_dec=True):
    """
    Function to fit the model to training set and print F1 scores for both training set
    and validation set.
    -------------------------------------------------------------------------------------
    Parameters:
        model: Model (Neural Network) to which Training set will fit
        lr: Learning rate (initil learning rate if annln=True)
        xtr: Input train array (for getting F1score on whole array)
        ytr: Output train array (for getting F1score on whole array)
        xvl: Input validation array (for val. F1score)
        yvl: Output validation array (for val. F1score)
        train_dl: Train DataLoader which loads training data in batches (should give Tensors as output)
        n_epochs: number of epochs
        loss: Loss function to calculate and backpropagate loss (eg: CrossEntropy)
        opt: Optimizer, to update weights (eg: RMSprop)
        metrics: Function to calculate score of model (eg: accuracy, F1 score)
        annln: (default=False) If to use LRAnnealing or not
        mult_dec: (default=True) If to dec. max Learning rate on every cosine cycle
    """
    if(annln): annl = lrAnnealing(lr, 40, 449, mult_dec)  # itr_per_epoch = len(xtrain) // batch_size
    for epoch in range(n_epochs):
        tl = iter(train_dl)
        length = len(train_dl)
        
        for t in range(length):
            xt, yt = next(tl)

            #y_pred = model(Variable(xt).cuda())
            #l = loss(y_pred, Variable(yt).cuda())
            y_pred = model(Variable(xt))
            l = loss(y_pred, Variable(yt))
            if(annln): annl(opt)
            opt.zero_grad()
            l.backward()
            opt.step()
        
        val_score = get_f1score(model, 
                                torch.DoubleTensor(np.array(xvl, dtype = "float32")), #.cuda
                                torch.LongTensor(np.array(yvl, dtype = "float32")-1)) #.cuda
        trn_score = get_f1score(model, 
                                torch.DoubleTensor(np.array(xtr, dtype = "float32")), #.cuda
                                torch.LongTensor(np.array(ytr, dtype = "float32")-1)) #.cuda
        
        if (epoch+1)%5 == 0:
            print("Epoch " + str(epoch) + "::"
                + "  trnF1score: " + str(trn_score)
                +", valF1score: " + str(val_score))
            
def get_f1score(model, x, y):
    """
    To get F1score of predictions from Neural Network.
    -----------------------------------------------------------------
    Parameters:
        model: Neural Network Model
        x: Input Values to be sent to model() function to get predictions
        y: Output Values to be checked with predictions
    Output:
        Return F1 score of predictions 
    """
    pred = model(Variable(x).contiguous())
    ypreds = np.argmax(pred.contiguous().data.numpy(), axis=1) #.cpu()
    yactuals = y.contiguous().numpy() #.cpu()
    return F1score(yactuals, ypreds)

def set_lr(opt, lr):
    """
    Function to set lr for optimizer in every layer.
    ------------------------------------------------------------------
    Parameters:
        opt: optimizer used in neural network
        lr: New Learning rate to be set in each layer
    """
    for pg in opt.param_groups: pg['lr'] = lr

class lrAnnealing():
    def __init__(self, ini_lr, epochs, itr_per_epoch, mult_dec):
        """
        Class to Anneal learning rate with warm restarts with time. It decreases 
        learning rate as multiple cosine waves with dec. amplitudes.1e-10 is taken 
        as zero. (The lower point for cosine)
        ---------------------------------------------------------------------------
        Parameters:
            ini_lr: Initial learning rate
            epochs: Number of epochs
            itr_per_epoch: iterations per epoch
            mult_dec: T/F, If to use Annealing with warm restarts or hard
        """
        self.epochs = epochs
        self.ipe = itr_per_epoch
        self.m_dec = mult_dec
        self.ppw = (self.ipe * self.epochs) // 4    # Points per wave of cosine (For 4 waves per fit method)
        self.count = 0
        self.lr = ini_lr
        self.values = np.cos(np.linspace(np.arccos(self.lr), np.arccos(1e-10), self.ppw))
        self.mult = 1
    def __call__(self, opt):
        """
            opt: optimizer of which lr is to set
        """
        self.count += 1
        set_lr(opt, self.values[self.count-1]*self.mult)
        if self.count == len(self.values):
            self.count = 0
            if(self.m_dec): self.mult /= 2
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
%time fit(net, 1e-2, xtrain, ytrain, xvalid, yvalid, trainLoader, 40, loss, opt, metrics, annln=False)
%time fit(net, 1e-3, xtrain, ytrain, xvalid, yvalid, trainLoader, 40, loss, opt, metrics, annln=False)
# Getting max F1 score of .85 in training and .85 for test
# But not giving good results on public leaderboard (0.303)
%time fit(net, 1e-4, xtrain, ytrain, xvalid, yvalid, trainLoader, 40, loss, opt, metrics, annln=False)
%time fit(net, 1e-5, xtrain, ytrain, xvalid, yvalid, trainLoader, 40, loss, opt, metrics, annln=False)
%time fit(net, 1e-6, xtrain, ytrain, xvalid, yvalid, trainLoader, 40, loss, opt, metrics, annln=False)
# Plotting the result:
ypreds = net(Variable(torch.DoubleTensor(np.array(xvalid, dtype="float32"))).contiguous()).data.numpy().argmax(1)+1 #.cuda, .cpu()
plot_bar_stacked(yvalid, ypreds)
#Ttrain = TensorDataset(torch.DoubleTensor(np.array(xtrain2.values, dtype="float32")), #.cuda
#                       torch.LongTensor(np.array(ytrain2.values, dtype="float32")-1)) #.cuda
#trainLoader = DataLoader(Ttrain, batch_size = 20, shuffle=True)
#net2 = Net(len(xtrain2.columns)).double() #.cuda()
#loss = nn.CrossEntropyLoss()
#metrics = [F1score]
#opt = optim.Adam(net.parameters(), weight_decay=1e-3)
#opt = optim.SGD(net.parameters(), 1e-3, momentum=0.999, weight_decay=1e-3, nesterov=True)
#opt = optim.RMSprop(net.parameters(), momentum=0.9, weight_decay=1e-3)
#ytrain2.unique(), yvalid2.unique()
#%time fit(net2, 1e-2, xtrain2, ytrain2, xvalid2, yvalid2, trainLoader, 40, loss, opt, metrics, annln=False)
#%time fit(net2, 1e-3, xtrain2, ytrain2, xvalid2, yvalid2, trainLoader, 40, loss, opt, metrics, annln=False)
#%time fit(net, 1e-4, xtrain2, ytrain2, xvalid2, yvalid2, trainLoader, 40, loss, opt, metrics, annln=False)
#%time fit(net, 1e-5, xtrain2, ytrain2, xvalid2, yvalid2, trainLoader, 40, loss, opt, metrics, annln=False)
# Plotting the result:
#ypreds = net2(Variable(torch.DoubleTensor(np.array(xvalid2, dtype="float32"))).contiguous()).data.numpy().argmax(1)+1 #.cuda, .cuda()
#plot_bar_stacked(yvalid2, ypreds)
m = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1)
m.fit(xtrain, ytrain)
print_score(m, (xtrain, ytrain), (xvalid, yvalid))

to_pred = test[xtrain.columns]
npArray = np.stack([lb3.inverse_transform(test['Id'].values), m.predict(to_pred)], axis=-1)

res = pd.DataFrame(npArray, index=np.arange(len(npArray)), columns=['Id', 'Target'])
res.to_csv("submission1.csv", index=False)
m2 = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1)
m2.fit(xtrain2, ytrain2)
to_pred = test[xtrain2.columns]
npArray = np.stack([lb3.inverse_transform(test['Id'].values), m2.predict(to_pred)], axis=-1)
res = pd.DataFrame(npArray, index=np.arange(len(npArray)), columns=['Id', 'Target'])
res.to_csv("submission2.csv", index=False)
# Uncomment it to get output for 3rd RandomForest Classifier (Ready for submission)
m3 = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1, class_weight='balanced')
m3.fit(xtrain3, ytrain3)
to_pred = test[xtrain3.columns]
npArray = np.stack([lb3.inverse_transform(test['Id'].values), m3.predict(to_pred)], axis=-1)

res = pd.DataFrame(npArray, index=np.arange(len(npArray)), columns=['Id', 'Target'])
res.to_csv("submission3.csv", index=False)
# For output of net (DNN-1 with all features)
pred = net(Variable(torch.DoubleTensor(np.array(test.drop(['Id', 'idhogar'], axis=1).values, dtype="float32"))).contiguous()) # .cuda
ypreds = pred.contiguous().data.numpy().argmax(1) + 1 # .cpu()
npArray = np.stack([lb3.inverse_transform(test['Id'].values),ypreds], axis=-1); npArray[0]

res = pd.DataFrame(npArray, index=np.arange(len(npArray)), columns=['Id', 'Target'])
res.to_csv("submission4.csv", index=False)
ypreds = classifier.predict(test[xtrain.columns]) + 1
npArray = np.stack([lb3.inverse_transform(test['Id'].values),ypreds], axis=-1)

res = pd.DataFrame(npArray, index=np.arange(len(npArray)), columns=['Id', 'Target'])
res.to_csv("submission5.csv", index=False)
ypreds = classifier2.predict(test[xtrain2.columns]) + 1
npArray = np.stack([lb3.inverse_transform(test['Id'].values),ypreds], axis=-1)
res = pd.DataFrame(npArray, index=np.arange(len(npArray)), columns=['Id', 'Target'])
res.to_csv("submission6.csv", index=False)