import pandas as pd
data = pd.read_csv('../input/diabetes.csv')
labels = ['low','medium','high']
for j in data.columns[:-1]:
    mean = data[j].mean()
    data[j] = data[j].replace(0,mean) #Replace 0 with mean
    data[j] = pd.cut(data[j],bins=len(labels),labels=labels)
def count(data,colname,label,target):
    condition = (data[colname] == label) & (data['Outcome'] == target)
    return len(data[condition])
predict = []
probabilities = {0:{},1:{}}
train_percent = 70
train_len = int((train_percent*len(data))/100)
train_X = data.iloc[:train_len,:]
test_X = data.iloc[train_len+1:,:-1]
test_y = data.iloc[train_len+1:,-1]
count_0 = count(train_X,'Outcome',0,0)
count_1 = count(train_X,'Outcome',1,1)
    
prob_0 = count_0/len(train_X)
prob_1 = count_1/len(train_X)
for col in train_X.columns[:-1]:
        probabilities[0][col] = {}
        probabilities[1][col] = {}
        
        for category in labels:
            count_ct_0 = count(train_X,col,category,0)
            count_ct_1 = count(train_X,col,category,1)
            
            probabilities[0][col][category] = count_ct_0 / count_0
            probabilities[1][col][category] = count_ct_1 / count_1
for row in range(0,len(test_X)):
        prod_0 = prob_0
        prod_1 = prob_1
        for feature in test_X.columns:
            prod_0 *= probabilities[0][feature][test_X[feature].iloc[row]]
            prod_1 *= probabilities[1][feature][test_X[feature].iloc[row]]
        
        #Predict the outcome
        if prod_0 > prod_1:
            predict.append(0)
        else:
            predict.append(1)
tp,tn,fp,fn = 0,0,0,0
for j in range(0,len(predict)):
    if predict[j] == 0:
        if test_y.iloc[j] == 0:
            tp += 1
        else:
            fp += 1
    else:
        if test_y.iloc[j] == 1:
            tn += 1
        else:
            fn += 1
print('Accuracy for training length '+str(train_percent)+'% : ',((tp+tn)/len(test_y))*100)