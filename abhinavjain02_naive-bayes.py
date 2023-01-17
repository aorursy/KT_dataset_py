import numpy as np
import pandas as pd
import math
train_df = pd.read_csv("../input/titanic/train.csv")
train_df.head()
num_samples = train_df.shape[0]
print("No of samples:",num_samples)
columns = list(train_df.keys())
print(columns)
for col in columns:
    missing = np.sum(train_df[col].isna())
    print(col," ",missing," ",(missing*100/train_df.shape[0]))
train_df['Age'] = train_df['Age'].fillna(np.mean(train_df['Age']))
for col in columns:
    missing = np.sum(train_df[col].isna())
    print(col," ",missing," ",(missing*100/train_df.shape[0]))
columns.remove('Cabin')
columns.remove('Survived')
columns.remove('Name')
columns.remove('PassengerId')
columns.remove('Ticket')
print(columns)
max_categorical_thresh = 3
domain = {}
isCategorical = {}
for attribute in columns:
    num_values = train_df[attribute].nunique()
    if num_values > max_categorical_thresh:
        isCategorical[attribute]=False
        domain[attribute] = 2
    else:
        isCategorical[attribute]=True
        domain[attribute]=num_values
print(domain)
print(isCategorical)
num_classes = 2
output_classes = [0,1]
df_by_attr = []
for attr in columns:
    if isCategorical[attr]:
        sub_df = train_df.groupby(['Survived',attr]).size().unstack(level=0)
        mask = (np.sum(np.array(sub_df == np.zeros(sub_df.shape[1])),axis=0,keepdims=True) > 0).astype('int32')
        sub_df+=mask
        sub_df/=np.sum(sub_df,axis=0)
        df_by_attr.append(sub_df)
    else:
        mean = train_df[['Survived',attr]].groupby('Survived').mean().rename(columns={attr:'mean'})
        std = train_df[['Survived',attr]].groupby('Survived').std().rename(columns={attr:'std'})
        df_by_attr.append(pd.concat([mean,std],axis=1).T)

# concatenate all sub dataframes
prob = pd.concat(df_by_attr,keys=columns)
print(prob)
prob_class = train_df['Survived'].value_counts()
prob_class /= np.sum(prob_class)
print(prob_class)
test_df = pd.read_csv('../input/titanic/test.csv')
test_df.head()
answer = {}
def normal(x,mean,std):
    y = 1.0/(std*np.sqrt(2*math.pi))
    y *= np.exp(-0.5*np.square((x-mean)/std))
    return y
for i in range(test_df.shape[0]):
    sample = test_df.iloc[i,:]
    label_prob=[]
    for label in output_classes:
        cur_prod = prob_class[label]
        mask = sample.isna()
        for col in columns:
            if mask[col]:
                continue
            if isCategorical[col]:
                cur_prod *= prob.loc[(col,sample[col]),label]
            else :
                cur_prod *= normal(sample[col],prob.loc[(col,'mean'),label],prob.loc[(col,'std'),label])
        label_prob.append(cur_prod)
    answer[sample['PassengerId']]=np.argmax(label_prob)
submission = pd.DataFrame(data={'PassengerId':test_df['PassengerId']})
submission['Survived']=submission['PassengerId'].apply(lambda x: answer[x])
submission.to_csv('submission.csv',index=False)
