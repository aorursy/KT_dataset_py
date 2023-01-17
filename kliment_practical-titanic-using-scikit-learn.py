import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
df.head(6)
def class_survival(Pclass):
    class_surv = 0
    total = 0
    classes = df[['Survived','Pclass']]
    for i, row in classes.iterrows():
        if classes['Pclass'][i] == Pclass:
            class_surv += classes['Survived'][i]
            total = total + 1
    surv_rate = class_surv/total
    print(f'Class {Pclass} survival rate: {surv_rate}')
    return surv_rate
class1_surival = class_survival(1)
class2_surival = class_survival(2)
class3_surival = class_survival(3)
def agefill(Pclass):
    agefill = df[['Age', 'Pclass']].copy()
    for i, row in agefill['Pclass'].iteritems():
        if agefill['Pclass'][i] != Pclass:
            agefill = agefill.drop(i)
    Pclass_mean = np.nanmean(agefill.iloc[:,0].copy())
    return Pclass_mean
ones_mean = agefill(1)
twos_mean = agefill(2)
threes_mean = agefill(3)
print(f'Class 1 avg age: {ones_mean}\nClass 2 avg age: {twos_mean}\nClass 3 avg age: {threes_mean}')
malesurv, males, females, femalesurv = [0 for _ in range(4)]

gend = df[['Survived','Sex']]
for i1, row1 in gend.iterrows():
    if gend['Sex'][i1] == str('male'):
        malesurv += gend['Survived'][i1]
        males = males + 1
    else:
        femalesurv += gend['Survived'][i1]
        females = females + 1
print(f'Female survival rate: {femalesurv/females}\nMale survival rate: {malesurv/males}')
last_names = df['Name'].str.split(",", expand=True)[0]
is_balkan = list()
for name in range(len(last_names)):
    if last_names[name][-2:] != 'ic' and last_names[name][-2:] != 'ff' and last_names[name][-2:] != 'ulos':
        is_balkan.append(0) 
    else:
       is_balkan.append(1)
print(f'There were (at least) {np.sum(is_balkan)} Southeastern Europeans in the training set.')
titles = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
titles_investigate = pd.DataFrame(data=(df['Survived'],titles)).transpose()
types_titles = pd.DataFrame(data=set( val for dic in titles for val in titles.values)) #17 types of titles

def title_survive(name_title):
    name_title_survive = 0
    name_title_total = 0
    survival_rate = 0
    for title, rowz in titles_investigate.iterrows():
        if titles_investigate.iloc[title,1] == name_title:
            name_title_total += 1
            name_title_survive += titles_investigate.iloc[title,0]
    survival_rate = name_title_survive / name_title_total
    return survival_rate

titles2 = []
for type_title in range(len(types_titles)):
    titles2.append(title_survive(types_titles[0][type_title]))
    print(f'{types_titles[0][type_title]} had survivability chance: {titles2[type_title]}')
embark = df[['Survived','Embarked']]
def embarkchance(port):
    port_embark = 0
    port_survive = 0
    for ie, rowe in embark.iterrows():
        if embark.iloc[ie,1] == port:
            port_embark += 1
            if embark.iloc[ie,0] == 1:
                port_survive += 1
    survival_rate = port_survive / port_embark
    return survival_rate
    
S_embark = embarkchance('S')
C_embark = embarkchance('C')
Q_embark = embarkchance('Q')
print(f'Embarking at Port S had survivability chance: {S_embark}\nEmbarking at Port C had survivability chance: {C_embark}\nEmbarking at Port Q had survivability chance: {Q_embark}')
# Age feature
agefill = df[['Age', 'Pclass']].copy()
for i, row in agefill.iterrows():
    if np.isnan(agefill['Age'][i]):
        if agefill['Pclass'].iloc[i] == 3:
            agefill['Age'].iloc[i] = threes_mean
        elif agefill['Pclass'].iloc[i] == 2:
            agefill['Age'].iloc[i] = twos_mean
        else:
            agefill['Age'].iloc[i] = ones_mean
    
agecol = agefill.iloc[:,0].copy()

# Gender feature
gendercol = df['Sex'].copy()
for i2, row2 in gendercol.iteritems():
    if gendercol[i2] == str('male'):
        gendercol[i2] = 1
    else:
        gendercol[i2] = 2

# Is_balkan feature 
# is_balkan

# Type of title associated
types_titles['Survival_Chance'] = titles2

titles3 = []
for title, row4 in titles.iteritems():
    for title_type, row5 in types_titles.iterrows():
        if titles[title] == types_titles.iloc[title_type,0]:
            titles3.append(types_titles.iloc[title_type,1])
            
# Port where passenger embarked
ports = []
for embark_port, rowport in embark.iterrows():
        if embark.iloc[embark_port,1] == 'S':
            ports.append(S_embark)
        elif embark.iloc[embark_port,1] == 'C':
            ports.append(C_embark)
        else:
            ports.append(Q_embark)
X = np.array(np.column_stack((df[['Pclass', 'Fare', 'Parch']],agecol, gendercol, is_balkan, titles3, ports))).copy()
y = np.array(df[['Survived']].copy()).ravel()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import KFold
kf = KFold(n_splits = 5)
for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

model = SVC(gamma=1, C=2)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

cm = confusion_matrix(y_test, y_predict)
acc = accuracy_score(y_predict, y_test)
print('CM:\n {}'.format( cm))
print('Accuracy: {}'.format(acc))
# For SVM, an example optimization (for demonstration purposes) would be

from sklearn.model_selection import GridSearchCV
parameters = {
    'gamma': [1, 2],
    'C': [1, 2],
    }
gs = GridSearchCV(model, parameters, cv=3)
gs.fit(X, y)
print(gs.best_params_)
sub_format = np.column_stack((df_real['PassengerId'].astype(int),y_predict_test.astype(int)))
df_submit = pd.DataFrame(data=sub_format)
df_submit.columns = ['PassengerId','Survived']
df_submit.to_csv(r'~\titanic\submit.csv', index=False)