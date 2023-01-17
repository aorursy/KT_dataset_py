import numpy as np
import pandas as pd
#Importing the libraries to visualize
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(11.7,8.27)})
#Creating the DataFrame
df = pd.read_csv('../input/car-consume/measurements.csv')
#Checking the head of the data
df.head()
#Creating a function to convert the values
def comma_converter(x):
    if type(x) == str and ',' in x:
        a = x.split(',')
        return float(a[0] + '.' + a[1])
    else:
        return float(x)
    
#Converting the data
df['distance'] = df['distance'].apply(comma_converter)
df['consume'] = df['consume'].apply(comma_converter)
df['temp_inside'] = df['temp_inside'].apply(comma_converter)
df['temp_outside'] = df['temp_outside'].apply(comma_converter)
df['refill liters'] = df['refill liters'].apply(comma_converter)
df['km_absolute'] = df['distance'].cumsum()
df['consume_liter'] = df['consume']/100*df['distance']
#Distance vs Km_Absolute
#How the the distances distributing during the full distance 
sns.lmplot(data=df,x='km_absolute',y='distance', hue='gas_type', height=8, aspect=1.7,)
plt.xlabel('Km_absolute [Km]')
plt.ylabel('distance [Km]')
#Consume vs Km_absolute
sns.lmplot(data = df, x = 'distance', y = 'consume', hue = 'gas_type', height=8,aspect=1.7)
plt.xlabel('distance [Km]')
plt.ylabel('consume [l/100km]')
sns.lmplot(data = df, x = 'km_absolute', y =  'temp_outside',hue='gas_type', height=8, aspect=1.7)
plt.xlabel('km_absolute [KM]')
plt.ylabel('temp_outside [°C]')
sns.lmplot(data=df, x = 'speed',y =  'consume',hue = 'gas_type', height=8, aspect=1.7)
plt.xlabel('speed [km/h]')
plt.ylabel('consume [l/100km]')
plt.figure(figsize=(16.7,8))
sns.scatterplot(df[df['refill liters']>0]['km_absolute'],df[df['refill liters']>0]['refill liters'])
plt.xlabel('km_absolute [Km]')
plt.ylabel('refill liters [l]')
km_e10 = df[df['gas_type'] == 'E10']['distance'].sum()
km_sp98 = df[df['gas_type'] == 'SP98']['distance'].sum()

avg_e10 = df[df['gas_type'] == 'E10']['distance'].mean()
avg_sp98 = df[df['gas_type'] == 'SP98']['distance'].mean()

std_e10 = df[df['gas_type'] == 'E10']['distance'].std()
std_sp98 = df[df['gas_type'] == 'SP98']['distance'].std()

print('DISTANCES:')
print('Kilometers with E10: ' + str(km_e10))
print('Kilometers with SP98: ' + str(km_sp98))
print('Average distance with E10: ' + str(avg_e10))
print('Average distance with SP98: ' + str(avg_sp98))
print('Deviation of the distances with E10: ' + str(std_e10))
print('Deviation of the distancees with SP98: ' + str(std_sp98))
print('\n')
print(100*'*')
print('\n')
print('SPEED:')
avg_sp_e10 = df[df['gas_type'] == 'E10']['speed'].mean()
avg_sp_sp98 = df[df['gas_type'] == 'SP98']['speed'].mean()

std_sp_e10 = df[df['gas_type'] == 'E10']['speed'].std()
std_sp_sp98 = df[df['gas_type'] == 'SP98']['speed'].std()

print('Average speed with E10: ' + str(avg_sp_e10))
print('Average speed with SP98: ' + str(avg_sp_sp98))
print('Deviation of the speed with E10: ' + str(std_sp_e10))
print('Deviation of the speed with SP98: ' + str(std_sp_sp98))
num = df['refill liters'].count()
num_list = []

for i in range(num):
    num_list.append(i)
num_list = num_list[::-1]
    
def refill_events(x):
    if x > 0:
        return num_list.pop()
    else:
        pass
#Crteating the event column   
df['refill_event'] = df['refill liters'].apply(refill_events)
#The na values have to be replaced with the last value
df['refill liters'].fillna(method = 'bfill', inplace = True)
df['refill_event'].fillna(method = 'bfill', inplace = True)
consume_per_refill = df.groupby('refill_event').agg({'consume_liter': sum, 'refill liters': np.mean, 'gas_type' : 'first', 'km_absolute': 'last','distance':sum})
consume_per_refill.sort_values('km_absolute', inplace= True)
consume_per_refill
consume_per_refill.round(2).plot('km_absolute',['consume_liter','refill liters'], kind='bar',figsize=(16.7,8))
plt.xlabel('km_absolute [Km]')
plt.ylabel('[liter]')
consume_per_refill['tank_inhalt'] = 0
consume_per_refill['tank_inhalt'] =-consume_per_refill['consume_liter']+consume_per_refill['refill liters']
consume_per_refill['tank_inhalt'] = consume_per_refill['tank_inhalt']
#The first fefill supopse to be 45l.
consume_per_refill['tank_inhalt'].loc[0] = 45
consume_per_refill['tank_inhalt'] = consume_per_refill['tank_inhalt'].cumsum()
consume_per_refill
plt.figure(figsize=(30,10))
plt.scatter(consume_per_refill['km_absolute'],consume_per_refill['consume_liter'],)
plt.scatter(consume_per_refill['km_absolute'],consume_per_refill['refill liters'],)
plt.scatter(consume_per_refill['km_absolute'],consume_per_refill['tank_inhalt'],)
plt.legend(['consumed liter','refilled liters','theorical tank inhalt'])
#Calculating the real consumption value
consume_per_refill['consume_from_refill'] = consume_per_refill['refill liters']/consume_per_refill['distance']*100
#Changing the nem of the consume_liter column because this value is the value from the board computer
consume_per_refill.rename(columns = {'consume_liter' : 'consumption_board_comp_l'}, inplace = True)
#Merging the aggregated table to the original df, then I have all calculetad value in the whole dataframe in each refill periods (events)
extended_df = df.merge(consume_per_refill['consumption_board_comp_l'], on='refill_event', how='outer')


#Calculating the ratio
extended_df['consume_ratio'] = extended_df['consume_liter']/extended_df['consumption_board_comp_l']

#Calculating the corrected value
extended_df['consumption_after_correction'] = extended_df['consume_ratio']*extended_df['refill liters']

#Checking the values
extended_df[['consume_liter','refill liters','consumption_board_comp_l','consume_ratio','consumption_after_correction']]

#Converting the values to [l/100km]
extended_df['consume_corrected'] = extended_df['consumption_after_correction']/extended_df['distance']*100
#checking the correction
extended_df.groupby('refill_event').agg({'consumption_after_correction' : sum, 'refill liters' : 'last'})
plt.figure(figsize=(22,6))
plt.ylim([0,10])
plt.plot(extended_df['consume_liter'].loc[1:],marker='*')
plt.plot(extended_df['consumption_after_correction'][1:], marker='*')
plt.legend(['consumed liter boardcomputer','consumed liter after correction'],loc = 'upper right')
sns.heatmap(extended_df.isnull(), cmap='winter', yticklabels = False, cbar = False)
#Function to change the gas_type column string values to integer, then they can be used in the ML section.
def gasType(x):
    if x == 'E10':
        return 0
    else:
        return 1

#Applying the function    
extended_df['gas_type_int'] = extended_df['gas_type'].apply(gasType)
#Just a few value is missing, I use the mean of the full column to fill the NaN values
fill_temp_inside = np.mean(extended_df['temp_inside'])
extended_df['temp_inside'].fillna(fill_temp_inside,inplace = True)

#Calculating a delta T column to see the difference between the inside and outer temperature
extended_df['delta_t'] = extended_df['temp_inside']-extended_df['temp_outside']
test_data = extended_df[extended_df['consume_corrected'].isnull()]
plt.figure(figsize=(30,6))
plt.plot(range(0,len(extended_df)),extended_df[['consume','consume_corrected']])
plt.ylim(0,20)
plt.xlabel('trip')
plt.ylabel('consumption [l/100km]')
plt.legend(['consume from boardcomputer','consume corrected'])
#Checking the values which event is that
extended_df[(extended_df['consume_corrected']>7.5) & (extended_df['consume_corrected'].shift(1)>7.5) \
            & (extended_df['consume_corrected'].shift(-1)>7.5)][['refill_event','consume_corrected']].groupby('refill_event').count()
extended_df = extended_df[~((extended_df['refill_event'] == 6) | (extended_df['refill_event'] == 7) | (extended_df['refill_event'] == 3) | (extended_df['refill_event'] == 4)) \
                          & ((extended_df['consume_corrected'] > 3) & (extended_df['consume_corrected'] < 8))]

plt.figure(figsize=(30,6))
plt.plot(range(0,len(extended_df)),extended_df[['speed']])
plt.xlabel('trip')
plt.ylabel('speed [km/h]')
print('AC is on ' + str(extended_df[extended_df['AC']>0]['consume'].count()) + 'x')
print('Sunny days ' + str(extended_df[extended_df['sun']>0]['consume'].count()) + 'x')
print('Rainy days ' + str(extended_df[extended_df['rain']>0]['consume'].count()) + 'x')
print('There are {} rows, have we after the data cleaning'.format(extended_df['consume'].count()))
print('The consumption with E10 is {}l'.format(round(extended_df[extended_df['gas_type_int'] == 0]['consume'].mean(),2)))
print('The consumption with SP98 is {}l'.format(round(extended_df[extended_df['gas_type_int'] == 1]['consume'].mean(),2)))
print('\n')
print('The consumption with E10 is {}l'.format(round(extended_df[extended_df['gas_type_int'] == 0]['consume_corrected'].mean(),2)))
print('The consumption with SP98 is {}l'.format(round(extended_df[extended_df['gas_type_int'] == 1]['consume_corrected'].mean(),2)))
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
plt.figure(figsize=(30,10))
sns.heatmap(extended_df.corr(),cmap='viridis')
X1 = extended_df[['speed','distance','temp_inside','gas_type_int','rain','sun','delta_t','AC']]
y1 = extended_df['consume']

X2 = extended_df[['speed','distance','temp_inside','gas_type_int','rain','sun','delta_t','AC']]
y2 = extended_df['consume_corrected']

X3 = extended_df[['speed','distance','temp_inside','gas_type_int','rain','sun','delta_t','AC']]
y3 = extended_df['consume']


X_train1,X_test1,y_train1,y_test1 = train_test_split(X1,y1, test_size=0.3, random_state = 101)
X_train2,X_test2,y_train2,y_test2 = train_test_split(X2,y2, test_size=0.3, random_state = 101)

X_test3 = test_data[['speed','distance','temp_inside','gas_type_int','rain','sun','delta_t','AC']]
y_test3 = test_data['consume']

X_train3 = extended_df[['speed','distance','temp_inside','gas_type_int','rain','sun','delta_t','AC']]
y_train3 = extended_df['consume']

X_train4 = extended_df[['speed','distance','temp_inside','gas_type_int','rain','sun','delta_t','AC']]
y_train4 = extended_df['consume_corrected']

lm1 = LinearRegression()
lm2 = LinearRegression()
lm3 = LinearRegression()
lm4 = LinearRegression()

lm1.fit(X_train1,y_train1)
lm2.fit(X_train2,y_train2)
lm3.fit(X_train3,y_train3)
lm4.fit(X_train4,y_train4)
a = pd.DataFrame(lm1.coef_,X1.columns,columns=['Original Consumption (splitted data)'])
b = pd.DataFrame(lm2.coef_,X2.columns,columns=['Corrected Consumption (splitted data)'])
c = pd.DataFrame(lm3.coef_,X3.columns,columns=['Original Consumption (full data)'])
# d = pd.DataFrame(lm4.coef_,X4.columns,columns=['Corrected Consumption(full data)'])

merge_b_c = b.merge(c,right_index=True, left_index=True)
# merge_b_c_d = merge_b_c.merge(d,right_index=True, left_index=True)
# summery = a.merge(merge_b_c_d,right_index=True, left_index=True)
# summery
pred1 = lm1.predict(X_test1)
pred2 = lm2.predict(X_test2)
pred3 = lm3.predict(X_test3)

plt.scatter(y_test1,pred1)
plt.scatter(y_test2,pred2)
plt.scatter(y_test3,pred3)
plt.legend(['a','b','c'])
sns.distplot(y_test1-pred1, bins=15)
sns.distplot(y_test2-pred2, bins=15)
sns.distplot(y_test3-pred3, bins=15)
plt.legend(['a','b','c'])
print('Original Consumption: ')
print('MAE: ' + str(metrics.mean_absolute_error(y_test1,pred1)))
print('MSE: ' + str(metrics.mean_squared_error(y_test1,pred1)))
print('RMSE: ' + str(np.sqrt(metrics.mean_squared_error(y_test1,pred1))))
print('R^2: ' + str(metrics.r2_score(y_test1,pred1)))
print('\n')

print('Corrected Consumption: ')
print('MAE: ' + str(metrics.mean_absolute_error(y_test2,pred2)))
print('MSE: ' + str(metrics.mean_squared_error(y_test2,pred2)))
print('RMSE: ' + str(np.sqrt(metrics.mean_squared_error(y_test2,pred2))))
print('R^2: ' + str(metrics.r2_score(y_test2,pred2)))
print('\n')

print('Original Consumption (full data): ')
print('MAE: ' + str(metrics.mean_absolute_error(y_test3,pred3)))
print('MSE: ' + str(metrics.mean_squared_error(y_test3,pred3)))
print('RMSE: ' + str(np.sqrt(metrics.mean_squared_error(y_test3,pred3))))
print('R^2: ' + str(metrics.r2_score(y_test3,pred3)))