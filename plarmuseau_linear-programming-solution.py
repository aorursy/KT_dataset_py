import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import xgboost as xgb 

from sklearn.metrics import r2_score



%matplotlib inline



from IPython.display import display, HTML

# Shows all columns of a dataframe

def show_dataframe(X, rows = 2):

    display(HTML(X.to_html(max_rows=rows)))



# Datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')    
#add X0-8 combinations

for xi in ['X1','X2','X3','X4','X5','X6','X8']:

    nieuwveld='X0'+xi

    train[nieuwveld]=train['X0']+'-'+train[xi]

    test[nieuwveld]=test['X0']+'-'+test[xi]



# Categorical features

cat_cols = []

for c in train.columns:

    if train[c].dtype == 'object':

        cat_cols.append(c)

print('Categorical columns:', cat_cols)



# Dublicate features

d = {}; done = []

cols = train.columns.values

for c in cols: d[c]=[]

for i in range(len(cols)):

    if i not in done:

        for j in range(i+1, len(cols)):

            if all(train[cols[i]] == train[cols[j]]):

                done.append(j)

                d[cols[i]].append(cols[j])

dub_cols = []

for k in d.keys():

    if len(d[k]) > 0: 

        # print k, d[k]

        dub_cols += d[k]        

print('Dublicates:', dub_cols)



# Constant columns

const_cols = []

for c in cols:

    if len(train[c].unique()) == 1:

        const_cols.append(c)

print('Constant cols:', const_cols)
# Glue train + test

train['eval_set'] = 0; test['eval_set'] = 1

df = pd.concat([train, test], axis=0, copy=True)

# Reset index

df.reset_index(drop=True, inplace=True)



def add_new_col(x):

    if x not in new_col.keys(): 

        # set n/2 x if is contained in test, but not in train 

        # (n is the number of unique labels in train)

        # or an alternative could be -100 (something out of range [0; n-1]

        return int(len(new_col.keys())/2)

    return new_col[x] # rank of the label



for c in cat_cols:

    # get labels and corresponding means

    new_col = train.groupby(c).y.quantile(q=0.05).sort_values().to_dict()

    df[c+'new'] = df[c].apply(add_new_col)



# show the result

#show_dataframe(df, 10)

print('Shape df',df.shape)

X = df.drop(list((set(const_cols) | set(dub_cols) | set(cat_cols))), axis=1)



# Train

X_train = X[X.eval_set == 0]

y_train = X_train.pop('y'); 

X_train = X_train.drop(['eval_set', 'ID'], axis=1)

show_dataframe(X_train, 10)

# Test

X_test = X[X.eval_set == 1]

X_test = X_test.drop(['y', 'eval_set', 'ID'], axis=1)



# Base score

y_mean = -y_train.median() #q0.01 LB 0.5546 q=0.25 0.5549

# Shapes



print('Shape X_train: {}\nShape X_test: {}'.format(X_train.shape, X_test.shape))
maxlen=1000   # change this limit ot 4000 if you have enough memory

#construct Matrix with slacks Z 

cat_cols_new = []

for c in cat_cols:

    temp=c+'new'

    cat_cols_new.append(temp)

print('New Categorical columns:', cat_cols_new)



Xc_train=X_train[:maxlen] #[cat_cols_new]

Xc_train['b_']=1.0

XXI_train = pd.concat([Xc_train.append(-Xc_train).reset_index(),   pd.DataFrame(np.identity(len(Xc_train)*2) ) ], axis=1)

X2_train=XXI_train*XXI_train



Zident=-pd.DataFrame(np.identity(len(Xc_train)),columns=[str(xz)+'z' for xz in range(len(Xc_train))] )

ZZident=Zident.append(Zident).reset_index()

XXIZZ_train= pd.concat([XXI_train,ZZident], axis=1)

print(XXIZZ_train.shape)

print(len(Xc_train)*3+16)

print(len(XXI_train)+len(Xc_train))

fv=pd.DataFrame( [0.0 for xi in range(len(Xc_train.columns)+len(Xc_train)*2+2)] + [1.0 for xi in range(len(Xc_train))],columns=['fv']) 



XXIZZ_train=XXIZZ_train.append(fv.set_index(XXIZZ_train.columns ).T).reset_index()



# the real y

b= -y_train[:maxlen].append(-y_train[:maxlen])

b= b.append(pd.DataFrame([0]) ) # find 0 point



# the minima from 'X0'

#b2 = -y_train[:maxlen].append(-X_train['X0new'][:maxlen])

#b2= b2.append(pd.DataFrame([0]) )



b2 = -X_train['X0X5new'][:maxlen].append(-X_train['X0X8new'][:maxlen])

b2= b2.append(pd.DataFrame([0]) )

#do the math



print(XXIZZ_train.shape)

#print(b.shape)



XXIZZ_train=XXIZZ_train.drop(['index','level_0'],axis=1)



# regression solution

solution=np.linalg.pinv(XXIZZ_train).dot(b)

# minimal solution

solution2=np.linalg.pinv(XXIZZ_train).dot(b2)



show_dataframe(pd.DataFrame(solution.T))

show_dataframe(pd.DataFrame(solution2.T))

show_dataframe(pd.DataFrame(y_train).T)

coeff=solution[:327]

coeff2=solution2[:327]

#print(coeff)

#print(coeff2)

print('slack',(solution[326:3016]).sum(),(solution[326:3016]*solution[326:3016]).sum())

print('slack2',(solution2[326:3016]).sum(),(solution2[326:3016]*solution2[326:3016]).sum())
Xc_test=X_test #[cat_cols_new]

Xc_test['b_']=1.0

Xd_train=X_train #[cat_cols_new]

Xd_train['b_']=1.0



y_pred=-Xc_test.dot(coeff2)

y_train_pred=-Xd_train.dot(coeff2)



plt.figure(figsize=(10,10))

sns.distplot(y_train, kde=False, color='g')

sns.distplot(y_pred, kde=False, color='b')

plt.title('Distr. of train and pred. test')



plt.figure(figsize=(10,10))

plt.title('True vs. Pred. train')

plt.plot([80,265], [80,265], color='g', alpha=0.3)

plt.scatter(x=y_train, y=y_train_pred, marker='.', alpha=0.5)

plt.scatter(x=[np.mean(y_train)], y=[np.mean(y_train_pred)], marker='o', color='red')

plt.xlabel('Real train'); plt.ylabel('Pred. train')



y_pr=pd.DataFrame(y_pred).reset_index()

y_pr.columns=['index','y']

test['y_pr']=y_pr['y']



output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': test['y_pr'] })

output.to_csv('submLP.csv', index=False)
