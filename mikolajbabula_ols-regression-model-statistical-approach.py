## import standard libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm
## import data and names files

data = pd.read_csv('../input/sample-dataset-for-ols-reg-model-statistics/imports-85.csv')

names = pd.read_csv('../input/sample-dataset-for-ols-reg-model-statistics/imports-85_names.csv',  sep='_')  ## default separator does not work
data.head(2)
names.values
header_names = ['Symboling', 'Normalized_losses', 'Brand', 'Fuel_type', 'Aspiration', 'Num_of_doors', 'Body_style', 'Drive_wheels', 'Engine_location', 'Wheel_base', 'Length', 'Width', 'Height', 'Curb_weight', 'Engine_type', 'Num_of_cylinders', 'Engine_size', 'Fuel_system', 'Bore', 'Stroke', 'Compression_ratio', 'Horsepower', 'Peak_rpm', 'City_mpg', 'Highway_mpg', 'Price']

data = pd.read_csv('../input/sample-dataset-for-ols-reg-model-statistics/imports-85.csv', names=header_names)

data.head(3)
def missing_table_creator(data):

    missing_values, number_of_missings, perc_of_missing, type_of_data = [], [], [], []

    m=0   

    for j in range(len(data)):

        for i in range(len(data.columns)):

            if data.values[j, i]=="?":

                missing_values.append(data.columns[i])             

    unique_list = np.unique(list(missing_values))

    for i in unique_list:

        number_of_missings.append(missing_values.count(i))

        perc_of_missing.append(str(np.round(int(number_of_missings[m])/int(len(data)) * 100, 2))+"%")

        m+=1

    for i in unique_list:

        type_of_data.append(type(data[i][0]))

    missings_table = {"Column's name": unique_list, "Number of missings": number_of_missings, "Percent of missings": perc_of_missing, "Type": type_of_data}

    missings_table = pd.DataFrame(missings_table)

    return missings_table
mst = missing_table_creator(data)

mst
np.unique(list(data.Normalized_losses))
normalized_losses_grouped = data.groupby(by='Normalized_losses').count()

normalized_losses_list = np.unique(normalized_losses_grouped.index)
for i in range(len(normalized_losses_list)):

    if normalized_losses_list[i]=="?":

        normalized_losses_list[i]=400
for i in range(len(normalized_losses_list)):

    normalized_losses_list[i]=int(normalized_losses_list[i])
norm_loss_sort = {"Args": normalized_losses_list, "Values": normalized_losses_grouped.values[:,0]}

norm_loss_sort = pd.DataFrame(norm_loss_sort)

norm_loss_sort = norm_loss_sort.sort_values("Args")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

ax1.title.set_text("Distribution of Normalized losses compared wtih normal distribution curve")

sns.distplot(normalized_losses_grouped.values[:,0], fit=norm, kde=True, ax=ax1, color='gray')

ax2.title.set_text("Plot of Normalized losses with N/A value changed into 333")

ax2.scatter(norm_loss_sort.values[:,0], norm_loss_sort.values[:,1], color='gray')
data.Normalized_losses = data.Normalized_losses.replace({"?": 400})
missing_table_creator(data)
## check for type of data

for i in missing_table_creator(data)["Column's name"]:

    print(data[str(i)].head(4))
np.unique(data.Num_of_doors)
data.Num_of_doors = data.Num_of_doors.replace({"four": 1, "two": 0})
data[data.Num_of_doors=="?"]
## sedans should have 4 doors

data.Num_of_doors = data.Num_of_doors.replace({"?": 1})
missing_table_creator(data)
bore_mean = np.round(data[data.Bore!="?"].Bore.astype("float32").mean(), 2)

data.Bore = data.Bore.replace({"?": bore_mean})
hpr_mean = np.round(data[data.Horsepower!="?"].Horsepower.astype("float32").mean(), 2)

data.Horsepower = data.Horsepower.replace({"?": hpr_mean})
peak_mean = np.round(data[data.Peak_rpm!="?"].Peak_rpm.astype("float32").mean(), 2)

data.Peak_rpm = data.Peak_rpm.replace({"?": peak_mean})
stroke_mean = np.round(data[data.Stroke!="?"].Stroke.astype("float32").mean(), 2)

data.Stroke = data.Stroke.replace({"?": stroke_mean})
price_mean = np.round(data[data.Price!="?"].Price.astype("float32").mean(), 2)

data.Price = data.Price.replace({"?": price_mean})
missing_table_creator(data)
## data without missing values

data.head()
## split data into train and labels sets

train = data.drop(columns='Price')

labels = data.Price
## name of columns for below dictionaries

names.values[50:80]
## dictionaries for categorical features

brand_dict = dict(zip(list(np.unique(data.Brand)), [2, 2, 2, 1, 2, 1, 0, 2, 1, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 1, 0, 2])) ##brands spltted into three categories: cheap, more expensive and premium

fuel_dict = dict(zip(list(np.unique(data.Fuel_type)), list(range(len(data.groupby(by=['Fuel_type']).count().iloc[:,0])))))

aspiration_dict = dict(zip(list(np.unique(data.Aspiration)), list(range(len(data.groupby(by=['Aspiration']).count().iloc[:,0])))))

body_dict = dict(zip(list(np.unique(data.Body_style)), [3, 1, 0, 2, 3]))

wheels_dict = dict(zip(list(np.unique(data.Drive_wheels)), list(range(len(data.groupby(by=['Drive_wheels']).count().iloc[:,0])))))

engine_loc_dict = dict(zip(list(np.unique(data.Engine_location)), list(range(len(data.groupby(by=['Engine_location']).count().iloc[:,0])))))

engine_type_dict = dict(zip(list(np.unique(data.Engine_type)), list(range(len(data.groupby(by=['Engine_type']).count().iloc[:,0])))))

cylinders_dict = dict(zip(list(np.unique(data.Num_of_cylinders)), [8, 5, 4, 6, 3, 12, 2]))

fuel_system_dict = dict(zip(list(np.unique(data.Fuel_system)), list(range(len(data.groupby(by=['Fuel_system']).count().iloc[:,0])))))
## mapping dictionaries and creating numerical values instead of categoricals

train["brand"] = data.Brand.map(brand_dict)

train["fuel_type"] = data.Fuel_type.map(fuel_dict)

train["aspiration"] = data.Aspiration.map(aspiration_dict)

train["body_style"] = data.Body_style.map(body_dict)

train["drive_wheels"] = data.Drive_wheels.map(wheels_dict)

train["engine_location"] = data.Engine_location.map(engine_loc_dict)

train["engine_type"] = data.Engine_type.map(engine_type_dict)

train["cylinders"] = data.Num_of_cylinders.map(cylinders_dict)

train["fuel_system"] = data.Fuel_system.map(fuel_system_dict)

train = train.drop(columns=['Brand', 'Fuel_type', 'Aspiration', 'Body_style', 'Drive_wheels', 'Engine_location', 'Engine_type', 'Num_of_cylinders', 'Fuel_system'])
import statsmodels.api as sm
x = sm.add_constant(train.astype('float64'))

x = np.array(x).astype("float64")

y = np.array(labels).astype("float64")

xnames = list(train.astype('float64').columns)

xnames.insert(0, 'Intercept')

first_model = sm.OLS(y, x)

first_results = first_model.fit()

print(first_results.summary(xname=xnames, yname='Price'))
from sklearn.model_selection import train_test_split
#n=int(input("Please enter n: "))

r_sq_mean = []

for i in range(150):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = sm.OLS(y_train, x_train)

    results = model.fit()

    pred = results.predict(x_test)

    r_sq = 1 - sum((pred - y_test)**2) / sum((y_test - y_test.mean())**2)

    r_sq_mean.append(r_sq)

print(np.mean(r_sq_mean))
corr = train.astype("float64").corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



cmap = sns.diverging_palette(220, 10, as_cmap=True)



f, ax = plt.subplots(figsize=(22, 22))

sns.heatmap(corr, cbar=True, annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.8, cbar_kws={"shrink": .5}, ax=ax)

ax.arrow(17, 15.5, -1, 0, head_width=0.5, head_length=0.7, width=.06, fc='b', ec='b')

ax.arrow(16, 14.5, -1, 0, head_width=0.5, head_length=0.7, width=.06, fc='b', ec='b')

ax.arrow(15, 13.5, -1, 0, head_width=0.5, head_length=0.7, width=.06, fc='b', ec='b')

ax.arrow(11.5, 10, 0, 1, head_width=0.5, head_length=0.7, width=.06, fc='b', ec='b')

ax.arrow(8, 6.5, -1, 0, head_width=0.5, head_length=0.7, width=.06, fc='b', ec='b')

ax.arrow(7, 5.5, -1, 0, head_width=0.5, head_length=0.7, width=.06, fc='b', ec='b')

ax.arrow(6, 4.5, -1, 0, head_width=0.5, head_length=0.7, width=.06, fc='b', ec='b')

ax.arrow(9, 7.5, -1, 0, head_width=0.5, head_length=0.7, width=.06, fc='b', ec='b')

ax.arrow(26, 24.5, -1, 0, head_width=0.5, head_length=0.7, width=.06, fc='b', ec='b')

ax.arrow(11, 9.5, -1, 0, head_width=0.5, head_length=0.7, width=.06, fc='b', ec='b')

ax.arrow(5, 3.5, -1, 0, head_width=0.5, head_length=0.7, width=.06, fc='b', ec='b')

ax.arrow(4, 2.5, -1, 0, head_width=0.5, head_length=0.7, width=.06, fc='b', ec='b')

ax.arrow(4, 2.5, -1, 0, head_width=0.5, head_length=0.7, width=.06, fc='b', ec='b')
x = sm.add_constant(train.drop(columns=['Num_of_doors', 'Wheel_base', 'Length', 'Width', 'Height', 'Curb_weight', 'Bore', 'Compression_ratio', 'Peak_rpm', 'City_mpg', 'Highway_mpg', 'fuel_system']).astype("float64"))

y = np.array(labels).astype("float64")

xnames = list(train.drop(columns=['Num_of_doors', 'Wheel_base', 'Length', 'Width', 'Height', 'Curb_weight', 'Bore', 'Compression_ratio', 'Peak_rpm', 'City_mpg', 'Highway_mpg', 'fuel_system']).astype("float64").columns)

xnames.insert(0, 'Intercept')

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

second_model = sm.OLS(y, x)

second_results = second_model.fit()

print(second_results.summary(xname=xnames, yname='Price'))
corr = train.drop(columns=['Num_of_doors', 'Wheel_base', 'Length', 'Width', 'Height', 'Curb_weight', 'Bore', 'Compression_ratio', 'Peak_rpm', 'City_mpg', 'Highway_mpg', 'fuel_system']).astype("float64").corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



cmap = sns.diverging_palette(220, 10, as_cmap=True)



f, ax = plt.subplots(figsize=(18, 18))

sns.heatmap(corr, cbar=True, annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=1.2, cbar_kws={"shrink": .5}, ax=ax)



ax.arrow(6, 4.5, -1, 0, head_width=0.3, head_length=0.4, width=.06, fc='b', ec='b')

ax.arrow(4.5, 3, 0, 1, head_width=0.3, head_length=0.4, width=.06, fc='b', ec='b')

ax.arrow(11, 9.5, -1, 0, head_width=0.3, head_length=0.4, width=.06, fc='b', ec='b')
corr = train.drop(columns=['drive_wheels', 'Horsepower', 'Num_of_doors', 'Wheel_base', 'Length', 'Width', 'Height', 'Curb_weight', 'Bore', 'Compression_ratio', 'Peak_rpm', 'City_mpg', 'Highway_mpg', 'fuel_system']).astype("float64").corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



cmap = sns.diverging_palette(220, 10, as_cmap=True)



f, ax = plt.subplots(figsize=(16, 16))

f.suptitle("Final plot of correlations. \n14 from 25 parameters have been deleted.", fontsize=20)

sns.heatmap(corr, cbar=True, annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=1.2, cbar_kws={"shrink": .5}, ax=ax)
train_after_del = train.drop(columns=['drive_wheels', 'Horsepower', 'Num_of_doors', 'Wheel_base', 'Length', 'Width', 'Height', 'Curb_weight', 'Bore', 'Compression_ratio', 'Peak_rpm', 'City_mpg', 'Highway_mpg', 'fuel_system']).astype("float64")
x = sm.add_constant(train_after_del)

y = np.array(labels).astype("float64")

xnames = list(train_after_del.columns)

xnames.insert(0, 'Intercept')

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

third_model = sm.OLS(y, x)

third_results = third_model.fit()

print(third_results.summary(xname=xnames, yname='Price'))
print("Despite we have deleted many features, the condition number is still too big:", np.linalg.cond(third_results.model.exog))
train_after_del.head()
## standarization for normalized losses column

norm_loss_std = (train_after_del.Normalized_losses - train_after_del.Normalized_losses.mean())/train_after_del.Normalized_losses.std()

train_after_del['Normalized_losses'] = train_after_del['Normalized_losses'].map(dict(zip(list(train_after_del.Normalized_losses), norm_loss_std)))

## standarization for engine size column

engine_size_std = (train_after_del.Engine_size - train_after_del.Engine_size.mean())/train_after_del.Engine_size.std()

train_after_del['Engine_size'] = train_after_del['Engine_size'].map(dict(zip(list(train_after_del.Engine_size), engine_size_std)))

## standarization for stroke column

stroke_std = (train_after_del.Stroke - train_after_del.Stroke.mean())/train_after_del.Stroke.std()

train_after_del['Stroke'] = train_after_del['Stroke'].map(dict(zip(list(train_after_del.Stroke), stroke_std)))

## standarization for cylinders column

cylinders_std = (train_after_del.cylinders - train_after_del.cylinders.mean())/train_after_del.cylinders.std()

train_after_del['cylinders'] = train_after_del['cylinders'].map(dict(zip(list(train_after_del.cylinders), cylinders_std)))

## standarization for symboling column

symboling_std = (train_after_del.Symboling - train_after_del.Symboling.mean())/train_after_del.Symboling.std()

train_after_del['Symboling'] = train_after_del['Symboling'].map(dict(zip(list(train_after_del.Symboling), symboling_std)))

## standarization for body style column

body_std = (train_after_del.body_style - train_after_del.body_style.mean())/train_after_del.body_style.std()

train_after_del['body_style'] = train_after_del['body_style'].map(dict(zip(list(train_after_del.body_style), body_std)))

## standarization for brand column

brand_std = (train_after_del.brand - train_after_del.brand.mean())/train_after_del.brand.std()

train_after_del['brand'] = train_after_del['brand'].map(dict(zip(list(train_after_del.brand), brand_std)))
train_after_del.head()
x = sm.add_constant(train_after_del)

y = np.array(labels).astype("float64")

xnames = list(train_after_del.columns)

xnames.insert(0, 'Intercept')

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

fourth_model = sm.OLS(y, x)

fourth_results = fourth_model.fit()

print(fourth_results.summary(xname=xnames, yname='Price'))
print("After standarizing we obtain the condition number:", np.round(np.linalg.cond(fourth_results.model.exog), 4), "and there is no warning regarding this parameter anymore.")
xnames = list(train.astype('float64').columns)

xnames.insert(0, 'Intercept')

alpha = 0.05

first_model_pv = {"Features": xnames, "p-value": np.round(first_results.pvalues, 6)} ## take features with p-values from the first model

first_model_pv = pd.DataFrame(first_model_pv) ## convert into df type

cols_to_delete = first_model_pv[first_model_pv["p-value"]>alpha] ## take exceeding p-values only

cols_to_delete
## dropping all above features

train_cols_del_pv = train.drop(columns=list(cols_to_delete.Features)[1:]).astype('float64')

train_cols_del_pv.head()
## standarization for normalized losses column

norm_loss_std = (train_cols_del_pv.Normalized_losses - train_cols_del_pv.Normalized_losses.mean())/train_cols_del_pv.Normalized_losses.std()

train_cols_del_pv['Normalized_losses'] = train_cols_del_pv['Normalized_losses'].map(dict(zip(list(train_cols_del_pv.Normalized_losses), norm_loss_std)))

## standarization for engine size column

engine_size_std = (train_cols_del_pv.Engine_size - train_cols_del_pv.Engine_size.mean())/train_cols_del_pv.Engine_size.std()

train_cols_del_pv['Engine_size'] = train_cols_del_pv['Engine_size'].map(dict(zip(list(train_cols_del_pv.Engine_size), engine_size_std)))

## standarization for stroke column

bore_std = (train_cols_del_pv.Bore - train_cols_del_pv.Bore.mean())/train_cols_del_pv.Bore.std()

train_cols_del_pv['Bore'] = train_cols_del_pv['Bore'].map(dict(zip(list(train_cols_del_pv.Bore), bore_std)))

## standarization for cylinders column

cylinders_std = (train_cols_del_pv.cylinders - train_cols_del_pv.cylinders.mean())/train_cols_del_pv.cylinders.std()

train_cols_del_pv['cylinders'] = train_cols_del_pv['cylinders'].map(dict(zip(list(train_cols_del_pv.cylinders), cylinders_std)))

## standarization for symboling column

peak_std = (train_cols_del_pv.Peak_rpm - train_cols_del_pv.Peak_rpm.mean())/train_cols_del_pv.Peak_rpm.std()

train_cols_del_pv['Peak_rpm'] = train_cols_del_pv['Peak_rpm'].map(dict(zip(list(train_cols_del_pv.Peak_rpm), peak_std)))

## standarization for body style column

body_std = (train_cols_del_pv.body_style - train_cols_del_pv.body_style.mean())/train_cols_del_pv.body_style.std()

train_cols_del_pv['body_style'] = train_cols_del_pv['body_style'].map(dict(zip(list(train_cols_del_pv.body_style), body_std)))

## standarization for brand column

brand_std = (train_cols_del_pv.brand - train_cols_del_pv.brand.mean())/train_cols_del_pv.brand.std()

train_cols_del_pv['brand'] = train_cols_del_pv['brand'].map(dict(zip(list(train_cols_del_pv.brand), brand_std)))
x = sm.add_constant(train_cols_del_pv)

x = np.array(x).astype("float64")

y = np.array(labels).astype("float64")

xnames = list(train.drop(columns=list(cols_to_delete.Features)[1:]).columns)

xnames.insert(0, 'Intercept')

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model_due_to_pv = sm.OLS(y, x)

results_due_to_pv = model_due_to_pv.fit()

print(results_due_to_pv.summary(xname=xnames, yname='Price'))
corr1 = train_cols_del_pv.corr()

mask1 = np.zeros_like(corr1, dtype=np.bool)

mask1[np.triu_indices_from(mask1)] = True



corr2 = train_after_del.corr()

mask2 = np.zeros_like(corr2, dtype=np.bool)

mask2[np.triu_indices_from(mask2)] = True



cmap = sns.diverging_palette(220, 10, as_cmap=True)



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 18))

sns.heatmap(corr1, cbar=True, annot=True, mask=mask1, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=1.2, cbar_kws={"shrink": .5}, ax=ax1)

ax1.title.set_text("OLS Model: columns deleted due to p-value\n11 from 25 variables left\n R-squared: 0.854")

sns.heatmap(corr2, cbar=True, annot=True, mask=mask1, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=1.2, cbar_kws={"shrink": .5}, ax=ax2)

ax2.title.set_text("OLS Model: columns deleted due colinearity\n11 from 25 variables left\n R-squared: 0.849")
#n=int(input("Please enter n: \n"))

n=70

x1 = sm.add_constant(train_cols_del_pv)

x1 = np.array(x1).astype("float64")

x2 = sm.add_constant(train_after_del)

x2 = np.array(x2).astype("float64")

y = np.array(labels).astype("float64")

global_mean = [[], []]

for j in range(n):

    r_sq_mean = [[],[]]

    for i in range(500):

        x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y, test_size=0.2)

        x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y, test_size=0.2)

        model1 = sm.OLS(y1_train, x1_train)

        model2 = sm.OLS(y2_train, x2_train)

        results1 = model1.fit()

        results2 = model2.fit()

        pred1 = results1.predict(x1_test)

        pred2 = results2.predict(x2_test)

        r_sq1 = 1 - sum((pred1 - y1_test)**2) / sum((y1_test - y1_test.mean())**2)

        r_sq2 = 1 - sum((pred2 - y2_test)**2) / sum((y2_test - y2_test.mean())**2)

        r_sq_mean[0].append(r_sq1)

        r_sq_mean[1].append(r_sq2)

    if j%4==0:

        print("Epoch", j+1, "- R-sqared for variables selected due to p-value:", np.round(np.mean(r_sq_mean[0]), 5), "- R-squared for variables selected due to colinearity:", np.round(np.mean(r_sq_mean[1]), 5))

    global_mean[0].append(np.round(np.mean(r_sq_mean[0])-np.mean(r_sq_mean[1]), 5))

for i in range(len(global_mean[0])):

    if global_mean[0][i]>=0:

        global_mean[1].append(1)

    else:

        global_mean[1].append(0)

global_mean = np.array(global_mean)

print("\nModel with variables selected due to p-value was better in", np.round(np.sum(global_mean[1])/n * 100, 2), "% during", n, "epochs.")

print("\nAvearge difference between two r-square parameters is", np.round(np.mean(global_mean[0]), 5))
print("Error term for the first approach:" ,np.round(fourth_results.resid.mean(), 10))

print("Error term for the second approach:" ,np.round(results_due_to_pv.resid.mean(), 10))
from statsmodels.stats.diagnostic import normal_ad, lilliefors

from statsmodels.stats.stattools import jarque_bera

from scipy.stats import shapiro, kstest
anderson_darling_test1 = normal_ad(fourth_results.resid)

shapiro_wilk_test1 = shapiro(fourth_results.resid)

lilliefors_test1 = lilliefors(fourth_results.resid)

kolmogorov_smirnov_test1 = kstest(fourth_results.resid, 'norm')

jarque_bera_test1 = jarque_bera(fourth_results.resid)[:2]



arr1 = np.array(anderson_darling_test1+shapiro_wilk_test1+lilliefors_test1+kolmogorov_smirnov_test1+jarque_bera_test1).reshape(5,2)

residual_results_table1 = {"Test statistic": arr1[:,0], "P-value": np.round(arr1[:,1], 6)}

residual_results_df1 = pd.DataFrame(residual_results_table1, index=["Anderson-Darlin", "Shapiro-Wilk", "Lilliefors", "Kolmogorov-Smirnov", "Jarque-Bera"])

residual_results_df1
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

res1 = fourth_results.resid

res2 = results_due_to_pv.resid

sm.qqplot(res1, line='s', ax=ax1)

ax1.title.set_text("Q-Q plot for the first approach")

sm.qqplot(res2, line='s', ax=ax2)

ax2.title.set_text("Q-Q plot for the second approach")

plt.show()
anderson_darling_test2 = normal_ad(results_due_to_pv.resid)

shapiro_wilk_test2 = shapiro(results_due_to_pv.resid)

lilliefors_test2 = lilliefors(results_due_to_pv.resid)

kolmogorov_smirnov_test2 = kstest(results_due_to_pv.resid, 'norm')

jarque_bera_test2 = jarque_bera(results_due_to_pv.resid)[:2]



arr2 = np.array(anderson_darling_test2+shapiro_wilk_test2+lilliefors_test2+kolmogorov_smirnov_test2+jarque_bera_test2).reshape(5,2)

residual_results_table2 = {"Test statistic": arr2[:,0], "P-value": np.round(arr2[:,1], 6)}

residual_results_df2 = pd.DataFrame(residual_results_table2, index=["Anderson-Darlin", "Shapiro-Wilk", "Lilliefors", "Kolmogorov-Smirnov", "Jarque-Bera"])

residual_results_df2
from statsmodels.stats.diagnostic import het_breuschpagan

from statsmodels.stats.diagnostic import het_goldfeldquandt

from statsmodels.stats.diagnostic import het_white
x = sm.add_constant(train_after_del)

x = np.array(x).astype("float64")

y = np.array(labels).astype("float64")

homo_breusche_pagan = het_breuschpagan(fourth_results.resid, x)

homo_goldfeld_quandt = het_goldfeldquandt(y, x, alternative='two-sided')

homo_white = het_white(fourth_results.resid, x)
arr3 = np.array(homo_breusche_pagan[:2]+homo_goldfeld_quandt[:2]+homo_white[:2]).reshape(3,2)

homo_results_table = {"Test statistic": arr3[:,0], "P-value": np.round(arr3[:,1], 6)}

homo_results_df = pd.DataFrame(homo_results_table, index=["Breusche-Pagan", "Goldfeld-Quandt", "White"])

homo_results_df
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 9))

ax1.scatter(fourth_results.predict(x), fourth_results.resid, c='gray', label='Residuals')

ax1.plot(np.linspace(50, 46450, 1000), np.zeros(1000), c='black', label="y=0 line")

ax1.title.set_text("Predictions vs. Residuals")

ax1.legend()

ax1.set_xlabel("Preditions")

ax1.set_ylabel("Residuals")

ax2.scatter(range(len(fourth_results.resid)), fourth_results.predict(x), c='g', label='Predictions')

ax2.scatter(range(len(fourth_results.resid)), y, c='b', label='True Values')

ax2.title.set_text("Prediction vs. True Values")

ax2.set_xlabel("Number of a sample")

ax2.set_ylabel("Price")

ax2.legend()
from statsmodels.stats.stattools import durbin_watson

from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox
durbin_watson_test = durbin_watson(fourth_results.resid)

breusch_godfrey_test = acorr_breusch_godfrey(fourth_results)[:2]

ljung_box_test = acorr_ljungbox(fourth_results.resid, boxpierce=True, lags=1)
dw_df = {"Test statistic": durbin_watson_test}

dw_df = pd.DataFrame(dw_df, index=["Durbin-Watson test"])

dw_df
bg_df = {"Test statistic": breusch_godfrey_test[0], "p-values": breusch_godfrey_test[1]}

bg_df = pd.DataFrame(bg_df, index=["Breusch–Godfrey test"])

bg_df
arr4 = np.array(ljung_box_test).reshape(2,2)



lj_df = {"Test statistic": arr4[:,0], "p-value": np.round(arr4[:,1], 8)}

lj_df = pd.DataFrame(lj_df, index=["Ljung-Box test", "Box-Pierce test"])

lj_df
from statsmodels.stats.diagnostic import linear_rainbow, linear_harvey_collier
rainbow = linear_rainbow(fourth_results)

rainbow_df = {"Test statistic": rainbow[0], "p-value": rainbow[1]}

rainbow_df = pd.DataFrame(rainbow_df, index=["Rainbow test"])

rainbow_df
harvey_collier = linear_harvey_collier(fourth_results)
from statsmodels.stats.outliers_influence import variance_inflation_factor
multicoll = [variance_inflation_factor(np.array(train_after_del), i) for i in range(train_after_del.shape[1])]

multicoll_df = {"Results": multicoll}

multicoll_df = pd.DataFrame(multicoll_df, index=train_after_del.columns)

multicoll_df
print("The interpretation is the following: for Engine size variable we have the variance inflation factor equal to", multicoll_df.Results[2], "taking the square root, we get, that this means that the standard error for the coefficient of that predictor variable is", np.round(np.sqrt(multicoll_df.Results[2]), 3), "times larger than if that predictor variable had 0 correlation with the other predictor variables." )
from statsmodels.stats.outliers_influence import variance_inflation_factor



multicoll2 = [variance_inflation_factor(np.array(train.astype('float64')), i) for i in range(train.shape[1])]

multicoll_df_ = {"All variables": multicoll2}

multicoll_df_ = pd.DataFrame(multicoll_df_, index=train.columns)

multicoll_df_
cond_num = {"1st Approach": np.linalg.cond(fourth_results.model.exog), "2nd Approach": np.linalg.cond(results_due_to_pv.model.exog), "All variables": np.linalg.cond(first_results.model.exog)}

cond_num = pd.DataFrame(cond_num, index=["Condition Number"])

cond_num
n=251

plt.figure(figsize=(35,15))



for i in train_after_del.columns:

    plt.subplot(n)

    n = n + 1

    x = train_after_del[str(i)]

    x = np.array(x)

    y = labels.astype('float64')

    z = np.linspace(np.min(x)-0.5, np.max(x)+0.5, 1000)

    line = fourth_results.params[0] + z*fourth_results.params[str(i)]

    plt.scatter(x, y)

    plt.plot(z, line)

    plt.title(str(i))

    if n>255:

        break

plt.show()



n=251

plt.figure(figsize=(35,15))



for i in train_after_del.columns[5:]:

    plt.subplot(n)

    n = n + 1

    x = train_after_del[str(i)]

    x = np.array(x)

    y = labels.astype('float64')

    z = np.linspace(np.min(x)-0.5, np.max(x)+0.5, 1000)

    line = fourth_results.params[0] + z*fourth_results.params[str(i)]

    plt.scatter(x, y)

    plt.plot(z, line)

    plt.title(str(i))

    if n>255:

        break

plt.show()
from statsmodels.stats.outliers_influence import OLSInfluence

from statsmodels.graphics.regressionplots import plot_leverage_resid2

from yellowbrick.regressor import CooksDistance
x = sm.add_constant(train_after_del)

x_arr = np.array(x).astype("float64")

y = np.array(labels).astype("float64")

list_=[]



f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 9))



x_train, x_test, y_train, y_test = train_test_split(x_arr, y, test_size=0.2)

final_model = sm.OLS(y_train, x_train)

xnames = list(train_after_del.astype('float64').columns)

xnames.insert(0, 'Intercept')

final_results = final_model.fit(xname=xnames, yname='Price')

ols_inf = OLSInfluence(final_results)

pred = final_results.predict(x_test)

r_sq = 1 - sum((pred - y_test)**2) / sum((y_test - y_test.mean())**2)

ax1.plot(range(len(pred)), np.sort(pred), c='g', label='Predictions')

ax1.plot(range(len(pred)), np.sort(y_test), c='b', label='True Values')

ax1.set_xlabel("Sorted samples")

ax1.set_ylabel("Price")

ax1.legend()

plot_leverage_resid2(final_results, ax=ax2)



visualizer = CooksDistance()

visualizer.fit(x_train, y_train)

visualizer.show(ax=ax3)



print("Final linear function is following:")

for i in range(len(final_results.params)):

    list_.append(str(np.round(final_results.params[i])) + " * " + str(x.columns[i]))

print('The final formula for the fitted line is:\n\n', '  +  '.join([lst for lst in list_]))

print("\nTest R-square:", r_sq)

print("\nCondition number:", np.linalg.cond(final_results.model.exog), '\n')

print(final_results.summary())

print("\nIn the below one can find another parameters that can be found in OLS Influence summary table:\n")

print(ols_inf.summary_table())