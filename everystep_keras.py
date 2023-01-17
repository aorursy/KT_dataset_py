import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam, SGD
class survivor_finder():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(32,activation='relu',input_shape=(5,)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64,activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32,activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1,activation='sigmoid'))
        
    def pretreatment(self,data):
        # 이용하지 않는 column 제거
        pop_col = ['PassengerId','Name','Ticket','Cabin','Embarked','Fare']
        for col in pop_col:
            data.pop(str(col))
        
        # Training Data 와 Test Data를 구분하여 전처리 진행
        if 'Survived' in data.keys():
            survived = data.pop('Survived')
            survived = survived.astype(np.float32)
        else:
            survived = None
        
        # 성별을 0과 1로 Mapping
        data['Sex'] = data['Sex'].map({'male':0,'female':1})
        # Age column의 NaN 칸을 Median으로 채움
        data['Age'].fillna(data['Age'].median(),inplace=True)

        # input = data / ouput = survived
        return data, survived
    
    def training(self,x,y):
        Loss = 'binary_crossentropy'
        self.model.compile(loss=Loss,optimizer=Adam(),metrics=['accuracy'])
        self.history = self.model.fit(x,y,batch_size=64,epochs=1000,verbose=1)
    
    def predict(self,data):
        prediction = np.array(self.model.predict(data,batch_size=16,verbose=0))
        
        # 우변의 기준값을 조정하여 예측할 생존자 수를 조정
        prediction = prediction > 0.6
        
        # Prediction에 True/False value를 1/0으로 변환
        prediction = prediction.astype(np.int)

        return prediction.T[0]
if __name__ == "__main__":
    #### file 경로 설정 ####
    import os
    dir_name = "../input"
    [train_file, __, test_file] = os.listdir(dir_name)
    train_file = os.path.join(dir_name,train_file)
    test_file = os.path.join(dir_name,test_file)
    
    #### Classifier Instance 생성 ####
    survivor_finder = survivor_finder()

    #### Training ####
    train_data = pd.read_csv(train_file)
    train_x, train_y = survivor_finder.pretreatment(train_data)
    survivor_finder.training(train_x,train_y)

    #### Test ####
    test_data = pd.read_csv(test_file)
    id = test_data["PassengerId"]
    test_x, __ = survivor_finder.pretreatment(test_data)
    prediction = survivor_finder.predict(test_x)
from keras.utils import normalize

data['Age'] = pd.Series(normalize(data['Age'].values)[0])