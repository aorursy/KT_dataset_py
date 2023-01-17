%matplotlib inline
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)

pd.set_option('display.float_format', lambda x: '%.2f' % x)
file_paths = list(map(lambda x: "../input/" + x + ".csv", ["app_events", "app_labels", "events", "gender_age_train", "label_categories"]))
 
data_sets = list(map(pd.read_csv, file_paths))
list(map(lambda x: x.shape, data_sets))
for data_set in data_sets:
    print(data_set.head())
apps = data_sets[0]['app_id']
print(apps.nunique())
value_count_app = apps.value_counts()
print(value_count_app.describe())
threshold = 200
unpop_apps = sns.kdeplot(value_count_app[value_count_app < threshold].values, shade = True)
unpop_apps.set_title('Density plot for apps of low popularity')
BAR_COUNT = 100
plt.figure()
ax = value_count_app.iloc[:BAR_COUNT].plot.bar()
ax.axes.get_xaxis().set_visible(False)
ax.set_title('Most popular apps')
pd.options.display.max_colwidth=80

label_cat = data_sets[4]
al = data_sets[1]

TOP_NUM = 5
top_apps = list(value_count_app.iloc[:TOP_NUM].index)
app_cat = pd.merge(al, label_cat, left_on = 'label_id', right_on = 'label_id').loc[:,['app_id','category']]
pd.concat([pd.DataFrame(app_cat[app_cat['app_id'] == app_id].groupby('app_id').aggregate(lambda x: tuple(x))['category'].values) 
           for app_id in top_apps])
gender_age = data_sets[3]
gender_value_counts = gender_age['gender'].value_counts(normalize = True)
gender_plot = gender_value_counts.plot.barh()
gender_plot.set_title('Gender frequency')
BINS = range(0,80,2)
gender_age_pivot = gender_age.loc[:,['gender','age']].pivot(columns = 'gender', values = 'age')

plt.figure()

age_female = gender_age_pivot['F'].plot.hist(normed = True, bins = BINS, alpha = 0.5)
age_female.set_title("Age distribution by gender")
age_male = gender_age_pivot['M'].plot.hist(normed = True, bins = BINS, alpha = 0.5)

female_patch = mpatches.Patch(color='blue', alpha = 0.5, label='female')
male_patch = mpatches.Patch(color='green', alpha = 0.5, label='male')
plt.legend(handles=[female_patch, male_patch])

plt.show()
events = data_sets[2]
TOP_APP_NUM = 10

device_events = pd.merge(gender_age, events, left_on = 'device_id', right_on = 'device_id').loc[:,['device_id','event_id']]
device_events_apps = pd.merge(device_events, data_sets[0], left_on = 'event_id', right_on = 'event_id').loc[:,['device_id','app_id']]
most_used = []
most_used_aux = device_events_apps.groupby('device_id').agg(lambda x: list(x.value_counts().index[0:TOP_APP_NUM]))['app_id']
most_used_aux.reset_index().apply(lambda row: [most_used.append([row['device_id'], app]) for app in row['app_id']], 
                                  axis=1)
most_used = pd.DataFrame(most_used, columns = ['device_id', 'app_id'])
gender_age_top_apps = pd.merge(gender_age, most_used, on = 'device_id')

top_female_apps = gender_age_top_apps[gender_age_top_apps['gender'] == 'F'].loc[:, 'app_id'].value_counts().index[0:TOP_APP_NUM]
top_male_apps = gender_age_top_apps[gender_age_top_apps['gender'] == 'M'].loc[:, 'app_id'].value_counts().index[0:TOP_APP_NUM]

pd.concat([pd.DataFrame(app_cat[app_cat['app_id'] == app_id].groupby('app_id').aggregate(lambda x: tuple(x))['category'].values) 
           for app_id in top_female_apps])
pd.concat([pd.DataFrame(app_cat[app_cat['app_id'] == app_id].groupby('app_id').aggregate(lambda x: tuple(x))['category'].values) 
           for app_id in top_male_apps])
gender_age_top_apps = pd.merge(gender_age, most_used, on = 'device_id')

top_young_apps = gender_age_top_apps[gender_age_top_apps['group'].isin(['M22-','F23-'])].loc[:, 'app_id'].value_counts().index[0:TOP_APP_NUM]
top_old_apps = gender_age_top_apps[gender_age_top_apps['group'].isin(['M39+','F43+'])].loc[:, 'app_id'].value_counts().index[0:TOP_APP_NUM]

pd.concat([pd.DataFrame(app_cat[app_cat['app_id'] == app_id].groupby('app_id').aggregate(lambda x: tuple(x))['category'].values) 
           for app_id in top_young_apps])
pd.concat([pd.DataFrame(app_cat[app_cat['app_id'] == app_id].groupby('app_id').aggregate(lambda x: tuple(x))['category'].values) 
           for app_id in top_old_apps])