import numpy as np
import pandas as pd

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
def refineDataSets(data, isTest):
    
    import numpy as np
    import pandas as pd
    import math 
    
    data = data.drop(['PassengerId', 'Ticket', 'Cabin', 'Fare'], axis=1)

    for i in range(0, data.shape[0]):
        if('Mr. ' in data['Name'][i]):
            data['Name'][i] = 0
            
        elif('Misc. ' in data['Name'][i]):
            data['Name'][i] = 1
            
        elif('Master. ' in data['Name'][i]):
            data['Name'][i] = 2
            
        elif('Miss. ' in data['Name'][i]):
            data['Name'][i] = 3
            
        elif('Mrs. ' in data['Name'][i]):
            data['Name'][i] = 4
            
        else:
            data['Name'][i] = 5
            
        if(data['Sex'][i] == 'female'):
            data['Sex'][i] = 1
        
        else:
            data['Sex'][i] = 0
    
        if(data['Embarked'][i] == 'S'):
            data['Embarked'][i] = 0
    
        elif(data['Embarked'][i] == 'C'):
            data['Embarked'][i] = 1
    
        else:
            data['Embarked'][i] = 2
            
        if(math.isnan(data['Age'][i])):
            data['Age'][i] = data['Age'].mean()
            
        if(math.isnan(data['Pclass'][i])):
            data['Pclass'][i] = data['Pclass'].mean()
            
        if(math.isnan(data['SibSp'][i])):
            data['SibSp'][i] = data['SibSp'].mean()
    
    
    data['Embarked'][0] = 0
    
    x_data = data.values
    x_data = np.float32(np.transpose(x_data))

    if isTest:
        return x_data
    
    p_data = np.array([data['Survived'].values])
    p_data = np.float32(np.transpose(p_data))
    return x_data, p_data

x_train_data, p_train_data = refineDataSets(train_data, False)
x_test_data = refineDataSets(test_data, True)
x_train_data = np.delete(x_train_data, 0, 0)
x_train_data = np.transpose(x_train_data)
x_test_data = np.transpose(x_test_data)
import sklearn
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1)).fit(np.concatenate((x_train_data, x_test_data), axis=0))

x_train_data = scaler.transform(x_train_data)
x_test_data = scaler.transform(x_test_data)
import ultimate
from ultimate.mlp import MLP

epoch_train = 1000

mlp = MLP(layer_size=[x_train_data.shape[1], 28, 28, 1], regularization=1, output_shrink=0.1, output_range=[0, 1], loss_type="hardmse")
mlp.train(x_train_data, p_train_data, iteration_log=20000, rate_init=0.08, rate_decay=0.8, epoch_train=epoch_train, epoch_decay=1)
predictions = np.round(mlp.predict(x_test_data).reshape(-1))
predictions = np.round(mlp.predict(x_test_data).reshape(-1))
ids = list(test_data['PassengerId'].values)
submission = pd.DataFrame(list(np.transpose([ids, predictions])))
submission.columns = ['PassengerId', 'Survived']
submission['Survived'] = np.int32(np.round(submission['Survived']))
submission['PassengerId'] = np.int32(np.round(submission['PassengerId']))
submission
submission.to_csv('titanic-submission-ultimate-nn-1.csv', index=False)