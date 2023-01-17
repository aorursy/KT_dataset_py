import pandas as pd
import numpy as np
df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')  
df
df.drop('date', axis=1, inplace=True)
df.drop('item_price', axis=1, inplace=True)
df
# Değerler 0-1 aralığına taşınıyor
df["shop_id"] = df["shop_id"]/59
df["item_id"] = df["item_id"]/22169

df
data = df.loc[:,["date_block_num", "shop_id", "item_id"]]
data = data.to_numpy()
labels = df["item_cnt_day"]
labels
labels = labels.to_numpy()
df2 = df.sample(n=1000000, random_state=1)
df2
data2 = df2.loc[:,["date_block_num", "shop_id", "item_id"]]
data2 = data2.to_numpy()
labels2 = df2["item_cnt_day"]
labels2
labels2 = labels2.to_numpy()
# Ağ üzerinden 3 adet nöron olduğu için 
# 8 adet ağırlık ve 3 adet bias değeri olmalı
global w11,w12,w13,w21,w22,w23,w31,w32,b1,b2,b3

## Sigmoid fonksiyonu 
def sigmoid(x):

    # Sigmoid aktivasyon fonksiyonu : f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

## Sigmoid fonksiyonunun türevi
def sigmoid_turev(x):

    # Sigmoid fonksiyonunun türevi: f'(x) = f(x) * (1 - f(x))
    sig = sigmoid(x)
    result = sig * (1 - sig)

    return result

def mse_loss(y_real, y_prediction):

    # y_real ve y_prediction aynı boyutta numpy arrayleri olmalıdır. 
    return ((y_real - y_prediction) ** 2).mean()

## İleri beslemeli nöronlar üzerinden tahmin
## değerinin elde edilmesi 

def feedforward(row):

    # h1 nöronunun değeri
    h1 = sigmoid((w11 * row[0]) + (w12 * row[1]) + (w13 * row[2]) + b1)

    # h2 nöronunun değeri
    h2 = sigmoid((w21 * row[0]) + (w22 * row[1]) + (w23 * row[2]) + b2)

    # Tahmin değeri o1 nöronun değeri
    o1 = sigmoid((w31 * h1) + (w32 * h2) + b3)

    return o1

## Belitiler iteresyon sayısı kadar (epochs) modeli eğitelim

def train(data, labels, epochs, learning_rate):
    
    global w11,w12,w13,w21,w22,w23,w31,w32,b1,b2,b3
    
    
    w11 = np.random.normal()
    w12 = np.random.normal()
    w13 = np.random.normal()    

    w21 = np.random.normal()
    w22 = np.random.normal()
    w23 = np.random.normal()
    
    w31 = np.random.normal()
    w32 = np.random.normal()
    

    b1 = np.random.normal()
    b2 = np.random.normal()
    b3 = np.random.normal()

    for epoch in range(epochs):

        for x, y in zip(data, labels):
            # Nöron H1
            sumH1 = (w11 * x[0]) + (w12 * x[1]) + (w13 * x[2]) + b1
            H1 = sigmoid(sumH1)

            # Nöron H2
            sumH2 = (w21 * x[0]) + (w22 * x[1]) + (w23 * x[2]) + b2
            H2 = sigmoid(sumH2)

            # Nöron O1
            sumO1 = (w31 * H1) + (w32 * H2) + b3
            O1 = sigmoid(sumO1)

            # Tahmin değerimiz
            prediction = O1

            # Türevlerin Hesaplanması
            # dL/dYpred :  y = doğru değer | prediciton: tahmin değeri
            dLoss_dPrediction = -2 * (y - prediction)

            # Nöron H1 için ağırlık ve bias türevleri 
            dH1_dW11 = x[0] * sigmoid_turev(sumH1)
            dH1_dW12 = x[1] * sigmoid_turev(sumH1)
            dH1_dW13 = x[2] * sigmoid_turev(sumH1)
            dH1_dB1 = sigmoid_turev(sumH1)

            # Nöron H2 için ağırlık ve bias türevleri
            dH2_dW21 = x[0] * sigmoid_turev(sumH2)
            dH2_dW22 = x[1] * sigmoid_turev(sumH2)
            dH2_dW23 = x[2] * sigmoid_turev(sumH2)                
            dH2_dB2 = sigmoid_turev(sumH2)

            # Nöron O1 (output) için ağırlık ve bias türevleri
            dPrediction_dW31 = H1 * sigmoid_turev(sumO1)
            dPrediction_dW32 = H2 * sigmoid_turev(sumO1)
            dPrediction_dB3 = sigmoid_turev(sumO1)

            # Aynı zamanda tahmin değerinin H1 ve H2'ye göre türevlerinin de
            # hesaplanması gerekmektedir. 
            dPrediction_dH1 = w31 * sigmoid_turev(sumO1)
            dPrediction_dH2 = w32 * sigmoid_turev(sumO1)

            ## Ağırlık ve biasların güncellenmesi 

            # H1 nöronu için güncelleme
            w11 = w11 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dW11)
            w12 = w12 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dW12)
            w13 = w13 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dW13)            
            b1  = b1 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dB1)

            # H2 nöronu için güncelleme 
            w21 = w21 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dW21)
            w22 = w22 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dW22)
            w23 = w23 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dW23)
            b2 = b2 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dB2)

            # O1 nöronu için güncelleme 
            w31 = w31 - (learning_rate * dLoss_dPrediction * dPrediction_dW31)
            w32 = w32 - (learning_rate * dLoss_dPrediction * dPrediction_dW32)
            b3 = b3 - (learning_rate * dLoss_dPrediction * dPrediction_dB3)

        predictions = np.apply_along_axis(feedforward, 1, data)
        loss = mse_loss(labels, predictions)
        
        #print("%d iterasyonda loss değeri: %.4f" % (epoch+1, loss))
        #print("%d iterasyonda: %.4f" % (epoch+1, loss))
    print("")
    print("%d iterasyonda loss değeri: %.4f" % (epoch+1, loss))
df_test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')  
df_test
df_test.drop('ID', axis=1, inplace=True)
df_test
x = np.arange(214200, dtype=int)
test_value = np.full_like(x, 34)
test_value = pd.DataFrame(test_value, columns = ['date_block_num'])
test_value
df_test = test_value.join(df_test)
df_test
df_test2 = df_test.sample(n=10000, random_state=1)
df_test2 = df_test.to_numpy()
df_test2
len(df_test2)
##### datamızı label'deki değerlere göre eğitiyoruz
print("eğitim basladi...")
train(data2, labels2, 100, 0.015)
print("eğitim tamamlandi...")
i=0
L = [];
tahmin = [];

while(i<len(df_test2)):
    
    A = [];
    L = df_test2[i]
    a1 = L[0]
    a2 = L[1]
    a3 = L[2]
    A.append(int(a1))
    A.append(int(a2)/59)
    A.append(int(a3)/22169)
    
    
    prediction = feedforward(A)
    
    if (prediction > 0.5):
        tahmin.append('1')
    else:
        tahmin.append('-1')
        
    #print(tahmin[i])
    i=i+1

tahminler = pd.DataFrame(tahmin, columns = ['item_cnt_day'])
result = df_test.join(tahminler)
result
