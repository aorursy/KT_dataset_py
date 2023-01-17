# Import Modules

import numpy as np
import pandas as pd
import os
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import os
os.chdir('../input/Data/Stocks/')
list = os.listdir()
number_files = len(list)
print(number_files)
#filenames = [x for x in os.listdir("./Stocks/") if x.endswith('.txt') and os.path.getsize(x) > 0]
filenames = random.sample([x for x in os.listdir() if x.endswith('.txt') and os.path.getsize(os.path.join('',x)) > 0], 20)
print(filenames)
df = []
for filename in filenames:
    dff = pd.read_csv(os.path.join('',filename), sep=',')
    label, _, _ = filename.split(sep='.')
    dff['Label'] = label
    dff['Date'] = pd.to_datetime(dff['Date'])
    df.append(dff)
df[0].head()
# We will consider the data with maximum entries among those 15 random datasets
len_of_data = []
for i in range(len(df)):
    len_of_data.append(len(df[i]))
print(max(len_of_data))

index = len_of_data.index(max(len_of_data))
print(index)

# Create 4 copies of data to add columns of different sets of Technical Indicators
data = df[index]
techindi1 = copy.deepcopy(data)
techindi2 = copy.deepcopy(data)
techindi3 = copy.deepcopy(data)
techindi4 = copy.deepcopy(data)
# Relative Strength Index
# Avg(PriceUp)/(Avg(PriceUP)+Avg(PriceDown)*100
# Where: PriceUp(t)=1*(Price(t)-Price(t-1)){Price(t)- Price(t-1)>0};
#        PriceDown(t)=-1*(Price(t)-Price(t-1)){Price(t)- Price(t-1)<0};

def rsi(values):
    up = values[values>0].mean()
    down = -1*values[values<0].mean()
    return 100 * up / (up + down)

# Add Momentum_1D column for all 15 stocks.
# Momentum_1D = P(t) - P(t-1)

techindi1['Momentum_1D'] = (techindi1['Close']-techindi1['Close'].shift(1)).fillna(0)
techindi1['RSI_14D'] = techindi1['Momentum_1D'].rolling(center=False, window=14).apply(rsi).fillna(0)
techindi1.tail(5)

techindi1['Volume_plain'] = techindi1['Volume'].fillna(0)
techindi1.tail()
def bbands(price, length=30, numsd=2):
    """ returns average, upper band, and lower band"""
    #ave = pd.stats.moments.rolling_mean(price,length)
    ave = price.rolling(window = length, center = False).mean()
    #sd = pd.stats.moments.rolling_std(price,length)
    sd = price.rolling(window = length, center = False).std()
    upband = ave + (sd*numsd)
    dnband = ave - (sd*numsd)
    return np.round(ave,3), np.round(upband,3), np.round(dnband,3)
techindi1['BB_Middle_Band'], techindi1['BB_Upper_Band'], techindi1['BB_Lower_Band'] = bbands(techindi1['Close'], length=20, numsd=1)
techindi1['BB_Middle_Band'] = techindi1['BB_Middle_Band'].fillna(0)
techindi1['BB_Upper_Band'] = techindi1['BB_Upper_Band'].fillna(0)
techindi1['BB_Lower_Band'] = techindi1['BB_Lower_Band'].fillna(0)
techindi1.tail()
def aroon(df, tf=25):
    aroonup = []
    aroondown = []
    x = tf
    while x< len(df['Date']):
        aroon_up = ((df['High'][x-tf:x].tolist().index(max(df['High'][x-tf:x])))/float(tf))*100
        aroon_down = ((df['Low'][x-tf:x].tolist().index(min(df['Low'][x-tf:x])))/float(tf))*100
        aroonup.append(aroon_up)
        aroondown.append(aroon_down)
        x+=1
    return aroonup, aroondown
listofzeros = [0] * 25
up, down = aroon(techindi1)
aroon_list = [x - y for x, y in zip(up,down)]
if len(aroon_list)==0:
    aroon_list = [0] * techindi1.shape[0]
    techindi1['Aroon_Oscillator'] = aroon_list
else:
    techindi1['Aroon_Oscillator'] = listofzeros+aroon_list

techindi1["PVT"] = (techindi1['Momentum_1D']/ techindi1['Close'].shift(1))*techindi1['Volume']
techindi1["PVT"] = techindi1["PVT"]-techindi1["PVT"].shift(1)
techindi1["PVT"] = techindi1["PVT"].fillna(0)
techindi1.tail()
def abands(df):
    #df['AB_Middle_Band'] = pd.rolling_mean(df['Close'], 20)
    df['AB_Middle_Band'] = df['Close'].rolling(window = 20, center=False).mean()
    # High * ( 1 + 4 * (High - Low) / (High + Low))
    df['aupband'] = df['High'] * (1 + 4 * (df['High']-df['Low'])/(df['High']+df['Low']))
    df['AB_Upper_Band'] = df['aupband'].rolling(window=20, center=False).mean()
    # Low *(1 - 4 * (High - Low)/ (High + Low))
    df['adownband'] = df['Low'] * (1 - 4 * (df['High']-df['Low'])/(df['High']+df['Low']))
    df['AB_Lower_Band'] = df['adownband'].rolling(window=20, center=False).mean()
abands(techindi1)
techindi1 = techindi1.fillna(0)
techindi1.tail()
columns2Drop = ['Momentum_1D', 'aupband', 'adownband']
techindi1 = techindi1.drop(labels = columns2Drop, axis=1)
techindi1.head()
def STOK(df, n):
    df['STOK'] = ((df['Close'] - df['Low'].rolling(window=n, center=False).mean()) / (df['High'].rolling(window=n, center=False).max() - df['Low'].rolling(window=n, center=False).min())) * 100
    df['STOD'] = df['STOK'].rolling(window = 3, center=False).mean()
STOK(techindi2, 4)
techindi2 = techindi2.fillna(0)
techindi2.tail()
def CMFlow(df, tf):
    CHMF = []
    MFMs = []
    MFVs = []
    x = tf
    
    while x < len(df['Date']):
        PeriodVolume = 0
        volRange = df['Volume'][x-tf:x]
        for eachVol in volRange:
            PeriodVolume += eachVol
        
        MFM = ((df['Close'][x] - df['Low'][x]) - (df['High'][x] - df['Close'][x])) / (df['High'][x] - df['Low'][x])
        MFV = MFM*PeriodVolume
        
        MFMs.append(MFM)
        MFVs.append(MFV)
        x+=1
    
    y = tf
    while y < len(MFVs):
        PeriodVolume = 0
        volRange = df['Volume'][x-tf:x]
        for eachVol in volRange:
            PeriodVolume += eachVol
        consider = MFVs[y-tf:y]
        tfsMFV = 0
        
        for eachMFV in consider:
            tfsMFV += eachMFV
        
        tfsCMF = tfsMFV/PeriodVolume
        CHMF.append(tfsCMF)
        y+=1
    return CHMF
listofzeros = [0] * 40
CHMF = CMFlow(techindi2, 20)
if len(CHMF)==0:
    CHMF = [0] * techindi2.shape[0]
    techindi2['Chaikin_MF'] = CHMF
else:
    techindi2['Chaikin_MF'] = listofzeros+CHMF
techindi2.tail()
def psar(df, iaf = 0.02, maxaf = 0.2):
    length = len(df)
    dates = (df['Date'])
    high = (df['High'])
    low = (df['Low'])
    close = (df['Close'])
    psar = df['Close'][0:len(df['Close'])]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af = iaf
    ep = df['Low'][0]
    hp = df['High'][0]
    lp = df['Low'][0]
    for i in range(2,length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
        reverse = False
        if bull:
            if df['Low'][i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = df['Low'][i]
                af = iaf
        else:
            if df['High'][i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = df['High'][i]
                af = iaf
        if not reverse:
            if bull:
                if df['High'][i] > hp:
                    hp = df['High'][i]
                    af = min(af + iaf, maxaf)
                if df['Low'][i - 1] < psar[i]:
                    psar[i] = df['Low'][i - 1]
                if df['Low'][i - 2] < psar[i]:
                    psar[i] = df['Low'][i - 2]
            else:
                if df['Low'][i] < lp:
                    lp = df['Low'][i]
                    af = min(af + iaf, maxaf)
                if df['High'][i - 1] > psar[i]:
                    psar[i] = df['High'][i - 1]
                if df['High'][i - 2] > psar[i]:
                    psar[i] = df['High'][i - 2]
        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]
    #return {"dates":dates, "high":high, "low":low, "close":close, "psar":psar, "psarbear":psarbear, "psarbull":psarbull}
    #return psar, psarbear, psarbull
    df['psar'] = psar
    #df['psarbear'] = psarbear
    #df['psarbull'] = psarbull
psar(techindi2)

techindi2.tail()
# ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100

techindi2['ROC'] = ((techindi2['Close'] - techindi2['Close'].shift(12))/(techindi2['Close'].shift(12)))*100
techindi2 = techindi2.fillna(0)
techindi2.tail()
techindi2['VWAP'] = np.cumsum(techindi2['Volume'] * (techindi2['High'] + techindi2['Low'])/2) / np.cumsum(techindi2['Volume'])
techindi2 = techindi2.fillna(0)
techindi2.tail()
techindi2['Momentum'] = techindi2['Close'] - techindi2['Close'].shift(4)
techindi2 = techindi2.fillna(0)
techindi2.tail()
def CCI(df, n, constant):
    TP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = pd.Series((TP - TP.rolling(window=n, center=False).mean()) / (constant * TP.rolling(window=n, center=False).std())) #, name = 'CCI_' + str(n))
    return CCI
techindi3['CCI'] = CCI(techindi3, 20, 0.015)
techindi3 = techindi3.fillna(0)
techindi3.tail()
new = (techindi3['Volume'] * (~techindi3['Close'].diff().le(0) * 2 -1)).cumsum()
techindi3['OBV'] = new
techindi3.tail()
#Keltner Channel  
def KELCH(df, n):  
    KelChM = pd.Series(((df['High'] + df['Low'] + df['Close']) / 3).rolling(window =n, center=False).mean(), name = 'KelChM_' + str(n))  
    KelChU = pd.Series(((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3).rolling(window =n, center=False).mean(), name = 'KelChU_' + str(n))  
    KelChD = pd.Series(((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3).rolling(window =n, center=False).mean(), name = 'KelChD_' + str(n))    
    return KelChM, KelChD, KelChU
KelchM, KelchD, KelchU = KELCH(techindi3, 14)
techindi3['Kelch_Upper'] = KelchU
techindi3['Kelch_Middle'] = KelchM
techindi3['Kelch_Down'] = KelchD
techindi3 = techindi3.fillna(0)
techindi3.tail()
techindi3['EMA'] = techindi3['Close'].ewm(span=3,min_periods=0,adjust=True,ignore_na=False).mean()
techindi3 = techindi3.fillna(0)


techindi3['TEMA'] = (3 * techindi3['EMA'] - 3 * techindi3['EMA'] * techindi3['EMA']) + (techindi3['EMA']*techindi3['EMA']*techindi3['EMA'])
techindi3.tail()
techindi3['HL'] = techindi3['High'] - techindi3['Low']
techindi3['absHC'] = abs(techindi3['High'] - techindi3['Close'].shift(1))
techindi3['absLC'] = abs(techindi3['Low'] - techindi3['Close'].shift(1))
techindi3['TR'] = techindi3[['HL','absHC','absLC']].max(axis=1)
techindi3['ATR'] = techindi3['TR'].rolling(window=14).mean()
techindi3['NATR'] = (techindi3['ATR'] / techindi3['Close']) *100
techindi3 = techindi3.fillna(0)
techindi3.tail()
def DMI(df, period):
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['Zero'] = 0

    df['PlusDM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > df['Zero']), df['UpMove'], 0)
    df['MinusDM'] = np.where((df['UpMove'] < df['DownMove']) & (df['DownMove'] > df['Zero']), df['DownMove'], 0)

    df['plusDI'] = 100 * (df['PlusDM']/df['ATR']).ewm(span=period,min_periods=0,adjust=True,ignore_na=False).mean()
    df['minusDI'] = 100 * (df['MinusDM']/df['ATR']).ewm(span=period,min_periods=0,adjust=True,ignore_na=False).mean()

    df['ADX'] = 100 * (abs((df['plusDI'] - df['minusDI'])/(df['plusDI'] + df['minusDI']))).ewm(span=period,min_periods=0,adjust=True,ignore_na=False).mean()
DMI(techindi3, 14)
techindi3 = techindi3.fillna(0)
techindi3.tail()
columns2Drop = ['UpMove', 'DownMove', 'ATR', 'PlusDM', 'MinusDM', 'Zero', 'EMA', 'HL', 'absHC', 'absLC', 'TR']

techindi3 = techindi3.drop(labels = columns2Drop, axis=1)
techindi3.head()
techindi4['26_ema'] = techindi4['Close'].ewm(span=26,min_periods=0,adjust=True,ignore_na=False).mean()
techindi4['12_ema'] = techindi4['Close'].ewm(span=12,min_periods=0,adjust=True,ignore_na=False).mean()
techindi4['MACD'] = techindi4['12_ema'] - techindi4['26_ema']
techindi4 = techindi4.fillna(0)
techindi4.tail()
def MFI(df):
    # typical price
    df['tp'] = (df['High']+df['Low']+df['Close'])/3
    #raw money flow
    df['rmf'] = df['tp'] * df['Volume']
    
    # positive and negative money flow
    df['pmf'] = np.where(df['tp'] > df['tp'].shift(1), df['tp'], 0)
    df['nmf'] = np.where(df['tp'] < df['tp'].shift(1), df['tp'], 0)

    # money flow ratio
    df['mfr'] = df['pmf'].rolling(window=14,center=False).sum()/df['nmf'].rolling(window=14,center=False).sum()
    df['Money_Flow_Index'] = 100 - 100 / (1 + df['mfr'])
MFI(techindi4)
techindi4 = techindi4.fillna(0)
techindi4.tail()
def ichimoku(df):
    # Turning Line
    period9_high = df['High'].rolling(window=9,center=False).max()
    period9_low = df['Low'].rolling(window=9,center=False).min()
    df['turning_line'] = (period9_high + period9_low) / 2
    
    # Standard Line
    period26_high = df['High'].rolling(window=26,center=False).max()
    period26_low = df['Low'].rolling(window=26,center=False).min()
    df['standard_line'] = (period26_high + period26_low) / 2
    
    # Leading Span 1
    df['ichimoku_span1'] = ((df['turning_line'] + df['standard_line']) / 2).shift(26)
    
    # Leading Span 2
    period52_high = df['High'].rolling(window=52,center=False).max()
    period52_low = df['Low'].rolling(window=52,center=False).min()
    df['ichimoku_span2'] = ((period52_high + period52_low) / 2).shift(26)
    
    # The most current closing price plotted 22 time periods behind (optional)
    df['chikou_span'] = df['Close'].shift(-22) # 22 according to investopedia
ichimoku(techindi4)
techindi4 = techindi4.fillna(0)
techindi4.tail()
def WillR(df):
    highest_high = df['High'].rolling(window=14,center=False).max()
    lowest_low = df['Low'].rolling(window=14,center=False).min()
    df['WillR'] = (-100) * ((highest_high - df['Close']) / (highest_high - lowest_low))

WillR(techindi4)
techindi4 = techindi4.fillna(0)
techindi4.tail()
def MINMAX(df):
    df['MIN_Volume'] = df['Volume'].rolling(window=14,center=False).min()
    df['MAX_Volume'] = df['Volume'].rolling(window=14,center=False).max()
MINMAX(techindi4)
techindi4 = techindi4.fillna(0)
techindi4.tail()
def KAMA(price, n=10, pow1=2, pow2=30):
    ''' kama indicator '''    
    ''' accepts pandas dataframe of prices '''

    absDiffx = abs(price - price.shift(1) )  

    ER_num = abs( price - price.shift(n) )
    ER_den = absDiffx.rolling(window=n,center=False).sum()
    ER = ER_num / ER_den

    sc = ( ER*(2.0/(pow1+1)-2.0/(pow2+1.0))+2/(pow2+1.0) ) ** 2.0


    answer = np.zeros(sc.size)
    N = len(answer)
    first_value = True

    for i in range(N):
        if sc[i] != sc[i]:
            answer[i] = np.nan
        else:
            if first_value:
                answer[i] = price[i]
                first_value = False
            else:
                answer[i] = answer[i-1] + sc[i] * (price[i] - answer[i-1])
    return answer
techindi4['KAMA'] = KAMA(techindi4['Close'])
techindi4 = techindi4.fillna(0)
techindi4.tail()
columns2Drop = ['26_ema', '12_ema','tp','rmf','pmf','nmf','mfr']

techindi4 = techindi4.drop(labels = columns2Drop, axis=1)
techindi4.head()
techindi1.index = techindi1['Date']
techindi1 = techindi1.drop(labels = ['Date'], axis = 1)

techindi2.index = techindi2['Date']
techindi2 = techindi2.drop(labels = ['Date'], axis = 1)

techindi3.index = techindi3['Date']
techindi3 = techindi3.drop(labels = ['Date'], axis = 1)

techindi4.index = techindi4['Date']
techindi4 = techindi4.drop(labels = ['Date'], axis = 1)
def normalized_df(df):
    normalized_df=(df-df.mean())/df.std()
    return normalized_df
normalized_df1 = copy.deepcopy(techindi1)
normalized_df2 = copy.deepcopy(techindi2)
normalized_df3 = copy.deepcopy(techindi3)
normalized_df4 = copy.deepcopy(techindi4)
ti_List1 = []
ti_List2 = []
ti_List3 = []
ti_List4 = []

x = normalized_df1['Label'][0]
ti_List1.append(x)
normalized_df1 = normalized_df1.drop('Label', 1)

x = normalized_df2['Label'][0]
ti_List2.append(x)
normalized_df2 = normalized_df2.drop('Label', 1)

x = normalized_df3['Label'][0]
ti_List3.append(x)
normalized_df3 = normalized_df3.drop('Label', 1)

x = normalized_df4['Label'][0]
ti_List4.append(x)
normalized_df4 = normalized_df4.drop('Label', 1)
normalized_df1.head()

mean = normalized_df1.mean(axis = 0)
normalized_df1 -= mean
std = normalized_df1.std(axis=0)
normalized_df1 /= std
    

mean = normalized_df2.mean(axis = 0)
normalized_df2 -= mean
std = normalized_df2.std(axis = 0)
normalized_df2 /= std
    

mean = normalized_df3.mean(axis = 0)
normalized_df3 -= mean
std = normalized_df3.std(axis = 0)
normalized_df3 /= std
    

mean = normalized_df4.mean(axis = 0)
normalized_df4 -= mean
std = normalized_df4.std(axis = 0)
normalized_df4 /= std

## Add the label class based on whether stock goes up or down
def add_label(df):
    idx = len(df.columns)
    new_col = np.where(df['Close'] >= df['Close'].shift(1), 1, 0)  
    df.insert(loc=idx, column='Label', value=new_col)
    df = df.fillna(0)

add_label(normalized_df1)    
add_label(normalized_df2)
add_label(normalized_df3)
add_label(normalized_df4)
normalized_df1.head()

normalized_df1 = normalized_df1.fillna(0)    
normalized_df2 = normalized_df2.fillna(0)
normalized_df3 = normalized_df3.fillna(0)
normalized_df4 = normalized_df4.fillna(0)
normalized_df1.head()

normalized_df1 = normalized_df1.values
normalized_df2 = normalized_df2.values
normalized_df3 = normalized_df3.values
normalized_df4 = normalized_df4.values
type(normalized_df1)
from keras.utils import to_categorical
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=32, step=5):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][-1]
        yield samples, to_categorical(targets)
# 10 10, 10, 64
# 5, 5, ,5 ,1
lookback = 5
step = 5
delay = 5
batch_size = 32
train_gen = generator(normalized_df1,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df1)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df1,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df1))+1,
                    max_index=round(0.8*len(normalized_df1)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df1,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df1))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df1)) - round(0.6*len(normalized_df1))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df1) - round(0.8*len(normalized_df1))+1 - lookback)
# How many steps to draw from test_gen in order to see the entire test set
a,b = next(train_gen)
# print labels
print(b)
from keras.models import Sequential 
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, normalized_df1.shape[-1]))) 
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=2, 
                              epochs=50, 
                              validation_data=val_gen, 
                              validation_steps=val_steps)
%matplotlib inline
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


test_loss, test_acc = model.evaluate_generator(test_gen, steps=3)
print('test acc:', test_acc)
print("test_loss:", test_loss)
train_gen = generator(normalized_df2,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df2)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df2,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df2))+1,
                    max_index=round(0.8*len(normalized_df2)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df2,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df2))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df2)) - round(0.6*len(normalized_df2))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df2) - round(0.8*len(normalized_df2))+1 - lookback)
model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, normalized_df2.shape[-1]))) 
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.summary()
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=2, 
                              epochs=5, 
                              validation_data=val_gen, 
                              validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
test_loss, test_acc = model.evaluate_generator(test_gen, steps=5)
print('test acc:', test_acc)
print("test_loss:", test_loss)
train_gen = generator(normalized_df3,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df3)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df3,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df3))+1,
                    max_index=round(0.8*len(normalized_df3)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df3,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df3))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df3)) - round(0.6*len(normalized_df3))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df3) - round(0.8*len(normalized_df3))+1 - lookback)
model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, normalized_df3.shape[-1]))) 
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.summary()
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=5, 
                              epochs=29, 
                              validation_data=val_gen, 
                              validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
test_loss, test_acc = model.evaluate_generator(test_gen, steps=4)
print('test acc:', test_acc)
print("test_loss:", test_loss)
train_gen = generator(normalized_df4,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df4)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df4,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df4))+1,
                    max_index=round(0.8*len(normalized_df4)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df4,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df4))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df4)) - round(0.6*len(normalized_df4))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df4) - round(0.8*len(normalized_df4))+1 - lookback)
model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, normalized_df4.shape[-1]))) 
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.summary()
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=5, 
                              epochs=30, 
                              validation_data=val_gen, 
                              validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
test_loss, test_acc = model.evaluate_generator(test_gen, steps=5)
print('test acc:', test_acc)
print("test_loss:", test_loss)
train_gen = generator(normalized_df1,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df1)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df1,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df1))+1,
                    max_index=round(0.8*len(normalized_df1)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df1,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df1))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df1)) - round(0.6*len(normalized_df1))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df1) - round(0.8*len(normalized_df1))+1 - lookback)
# How many steps to draw from test_gen in order to see the entire test set
model = Sequential()
model.add(layers.GRU(32, input_shape=(None, normalized_df1.shape[-1])))

model.add(layers.Dense(2, activation='softmax'))

model.summary()
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=2,
                              epochs=50,
                              validation_data=val_gen,
                             validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate_generator(test_gen, steps=1)
print('test acc:', test_acc)
print("test_loss:", test_loss)
train_gen = generator(normalized_df2,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df2)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df2,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df2))+1,
                    max_index=round(0.8*len(normalized_df2)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df2,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df2))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df2)) - round(0.6*len(normalized_df2))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df2) - round(0.8*len(normalized_df2))+1 - lookback)
model = Sequential()
model.add(layers.GRU(32, input_shape=(None, normalized_df2.shape[-1])))

model.add(layers.Dense(2, activation='softmax'))

model.summary()
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=2,
                              epochs=4,
                              validation_data=val_gen,
                             validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
test_loss, test_acc = model.evaluate_generator(test_gen, steps=2)
print('test acc:', test_acc)
print("test_loss:", test_loss)
train_gen = generator(normalized_df3,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df3)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df3,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df3))+1,
                    max_index=round(0.8*len(normalized_df3)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df3,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df3))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df3)) - round(0.6*len(normalized_df3))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df3) - round(0.8*len(normalized_df3))+1 - lookback)
model = Sequential()
model.add(layers.GRU(32, input_shape=(None, normalized_df3.shape[-1])))

model.add(layers.Dense(2, activation='softmax'))

model.summary()
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=2,
                              epochs=6,
                              validation_data=val_gen,
                             validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
test_loss, test_acc = model.evaluate_generator(test_gen, steps=7)
print('test acc:', test_acc)
print("test_loss:", test_loss)
train_gen = generator(normalized_df4,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df4)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df4,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df4))+1,
                    max_index=round(0.8*len(normalized_df4)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df4,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df4))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df4)) - round(0.6*len(normalized_df4))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df4) - round(0.8*len(normalized_df4))+1 - lookback)
model = Sequential()
model.add(layers.GRU(32, input_shape=(None, normalized_df4.shape[-1])))

model.add(layers.Dense(2, activation='softmax'))

model.summary()
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=2,
                              epochs=15,
                              validation_data=val_gen,
                             validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
test_loss, test_acc = model.evaluate_generator(test_gen, steps=7)
print('test acc:', test_acc)
print("test_loss:", test_loss)
train_gen = generator(normalized_df1,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df1)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df1,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df1))+1,
                    max_index=round(0.8*len(normalized_df1)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df1,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df1))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df1)) - round(0.6*len(normalized_df1))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df1) - round(0.8*len(normalized_df1))+1 - lookback)
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.4,
                     input_shape=(None, normalized_df1.shape[-1])))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=2,
                              epochs=50,
                              validation_data=val_gen,
                             validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate_generator(test_gen, steps=3)
print('test acc:', test_acc)
print("test_loss:", test_loss)
train_gen = generator(normalized_df2,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df2)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df2,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df2))+1,
                    max_index=round(0.8*len(normalized_df2)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df2,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df2))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df2)) - round(0.6*len(normalized_df2))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df2) - round(0.8*len(normalized_df2))+1 - lookback)
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.4,
                     input_shape=(None, normalized_df2.shape[-1])))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=2,
                              epochs=36,
                              validation_data=val_gen,
                             validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
test_loss, test_acc = model.evaluate_generator(test_gen, steps=4)
print('test acc:', test_acc)
print("test_loss:", test_loss)
train_gen = generator(normalized_df3,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df3)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df3,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df3))+1,
                    max_index=round(0.8*len(normalized_df3)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df3,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df3))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df3)) - round(0.6*len(normalized_df3))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df3) - round(0.8*len(normalized_df3))+1 - lookback)
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.4,
                     input_shape=(None, normalized_df3.shape[-1])))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=2,
                              epochs=20,
                              validation_data=val_gen,
                             validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
test_loss, test_acc = model.evaluate_generator(test_gen, steps=4)
print('test acc:', test_acc)
print("test_loss:", test_loss)
train_gen = generator(normalized_df4,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df4)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df4,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df4))+1,
                    max_index=round(0.8*len(normalized_df4)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df4,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df4))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df4)) - round(0.6*len(normalized_df4))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df4) - round(0.8*len(normalized_df4))+1 - lookback)
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.4,
                     input_shape=(None, normalized_df4.shape[-1])))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=2,
                              epochs=13,
                              validation_data=val_gen,
                             validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
test_loss, test_acc = model.evaluate_generator(test_gen, steps=7)
print('test acc:', test_acc)
print("test_loss:", test_loss)
train_gen = generator(normalized_df1,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df1)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df1,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df1))+1,
                    max_index=round(0.8*len(normalized_df1)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df1,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df1))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df1)) - round(0.6*len(normalized_df1))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df1) - round(0.8*len(normalized_df1))+1 - lookback)
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, normalized_df1.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=2,
                              epochs=5,
                              validation_data=val_gen,
                             validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate_generator(test_gen, steps=10)
print('test acc:', test_acc)
print("test_loss:", test_loss)
train_gen = generator(normalized_df2,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df2)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df2,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df2))+1,
                    max_index=round(0.8*len(normalized_df2)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df2,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df2))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df2)) - round(0.6*len(normalized_df2))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df2) - round(0.8*len(normalized_df2))+1 - lookback)
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, normalized_df2.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(2, activation = 'softmax'))
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=2,
                              epochs=10,
                              validation_data=val_gen,
                             validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate_generator(test_gen, steps=3)
print('test acc:', test_acc)
print("test_loss:", test_loss)
train_gen = generator(normalized_df3,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df3)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df3,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df3))+1,
                    max_index=round(0.8*len(normalized_df3)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df3,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df3))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df3)) - round(0.6*len(normalized_df3))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df3) - round(0.8*len(normalized_df3))+1 - lookback)
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, normalized_df3.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=2,
                              epochs=50,
                              validation_data=val_gen,
                             validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate_generator(test_gen, steps=5)
print('test acc:', test_acc)
print("test_loss:", test_loss)
train_gen = generator(normalized_df4,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=round(0.6*len(normalized_df4)),
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(normalized_df4,
                    lookback=lookback,
                    delay=delay,
                    min_index=round(0.6*len(normalized_df4))+1,
                    max_index=round(0.8*len(normalized_df4)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(normalized_df4,
                     lookback=lookback,
                     delay=delay,
                     min_index=round(0.8*len(normalized_df4))+1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (round(0.8*len(normalized_df4)) - round(0.6*len(normalized_df4))+1 - lookback) # how many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(normalized_df4) - round(0.8*len(normalized_df4))+1 - lookback)
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, normalized_df4.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=2,
                              epochs=50,
                              validation_data=val_gen,
                             validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
test_loss, test_acc = model.evaluate_generator(test_gen, steps=5)
print('test acc:', test_acc)
print("test_loss:", test_loss)