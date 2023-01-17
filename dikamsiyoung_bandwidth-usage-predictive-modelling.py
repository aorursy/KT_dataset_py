!conda install -y gdown
!gdown https://drive.google.com/uc?id=1ZZE3C866AtS9AyVOi3MNMQ9P3I0kwGIt
import pandas as pd

import numpy as np



df = pd.read_csv('/kaggle/working/online_meetings.csv')

df
df.isnull().sum()
df.dtypes
df.Platform.value_counts()
df.replace({'Zoom ':'zoom', 'Zoom':'zoom', 

            'Free Conference Call':'free_conf_call', 

            'Google Meet':'google_meet', 'Mixlr (audio)':'mixlr', 

            'Hangouts':'hangouts'}, inplace=True)
start_day = pd.DataFrame({'start_day': [pd.Timestamp(i).day for i in df['Start Time']]})

start_hr = pd.DataFrame({'start_hr':[pd.Timestamp(i).hour for i in df['Start Time']]})

start_min = pd.DataFrame({'start_min':[pd.Timestamp(i).minute for i in df['Start Time']]})
end_day = pd.DataFrame({'end_day':[pd.Timestamp(i).day for i in df['End Time']]})

end_hr = pd.DataFrame({'end_hr':[pd.Timestamp(i).hour for i in df['End Time']]})

end_min = pd.DataFrame({'end_min':[pd.Timestamp(i).minute for i in df['End Time']]})
duration_sec = pd.DataFrame({'duration_sec':[pd.Timedelta(i).seconds for i in df['Duration']]})
df_dummy = df
upload_speed = pd.DataFrame({'avg_upload_speed': [(b/s)*8 for b, s in zip(df.Upload, duration_sec.duration_sec)]})

download_speed = pd.DataFrame({'avg_download_speed': [(b/s)*8 for b, s in zip(df.Download, duration_sec.duration_sec)]})
df_dummy = pd.concat([df, start_day, start_hr, start_min, end_day, end_hr, end_min, duration_sec, download_speed, upload_speed], axis=1)

df_dummy.head()
import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set(style="whitegrid")
_ = sns.countplot(df_dummy.Platform)
_ = sns.scatterplot(df_dummy.duration_sec, df_dummy.Total,hue=df_dummy.Platform, palette='Set2')
_ = sns.scatterplot(df_dummy.duration_sec, df_dummy.Upload,hue=df_dummy['Participant Video on'], palette='Set2')
_ = sns.scatterplot(df_dummy.duration_sec, df_dummy.Upload,hue=df_dummy['Participant Screen Share'], palette='Set2')
_ = sns.scatterplot(df_dummy.duration_sec, df_dummy.Download,hue=df_dummy['Others Video on'], palette='Set2')
_ = sns.scatterplot(df_dummy.duration_sec, df_dummy.Download,hue=df_dummy['Others Screen Share'], palette='Set2')
_ = sns.scatterplot(df_dummy.duration_sec, df_dummy.Total,hue=df_dummy['Window Minimized'], palette='Set2')
def scatter(x,y, **kwargs):

    sns.scatterplot(x, y, palette='Set2')



g = sns.FacetGrid(df_dummy, col='Platform', row='Others Video on', hue='Participant Video on', aspect=1, palette='Set2')

_ = g.map(scatter, "duration_sec", "Total")

plt.subplots_adjust(hspace=0.5, wspace=0.3)

_ = g.add_legend()

_ = g.fig.set_size_inches(15,5)

_ = g.fig.suptitle("Video", y=1.08)

_ = g.set_titles(row_template = 'Others Vid {row_name}', col_template = '{col_name}')
def scatter(x,y, **kwargs):

    sns.scatterplot(x, y, palette='Set2')



g = sns.FacetGrid(df_dummy, col='Platform', row='Others Screen Share', hue='Participant Screen Share', aspect=1, palette='Set2')

_ = g.map(scatter, "duration_sec", "Total")

plt.subplots_adjust(hspace=0.5, wspace=0.3)

_ = g.fig.set_size_inches(15,5)

_ = g.add_legend()

_ = g.fig.suptitle("Screen Share", y=1.08)

_ = g.set_titles(row_template = 'Others Screen {row_name}', col_template = '{col_name}')
df_dummy.corr()
platform_dummy = pd.get_dummies(df_dummy.Platform, drop_first=True)

platform_dummy.head()
df_dummy = pd.concat([df_dummy, platform_dummy], axis=1)
df_dummy.dtypes
df_dummy.drop(['Start Time', 'End Time', 'Duration', 'Platform'], axis=1, inplace=True)

df_dummy.head()
X = df_dummy[df_dummy.zoom == 1].drop(['Download', 'Upload', 'Total', 'google_meet', 'mixlr', 'zoom', 'hangouts', 'Window Minimized'], axis=1)

X.head()
y = df_dummy.Total[df_dummy.zoom == 1]

y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=92)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

_ = regressor.fit(X_train, y_train)
regressor.score(X,y)
y_pred = regressor.predict(X_test)

y_pred = [5 if i < 0 else i for i in y_pred]
df_result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df_result.head()
df_result.head(30).plot(kind='bar', figsize=(20,8))
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(X.avg_upload_speed.mean())

print(X.avg_download_speed.mean())
is_group = 1

participant_video = 0

participant_mic = 1

participant_screen = 1

others_video = 0

others_screen = 0

start_hr = 12

start_min = 15

end_hr = 12

end_min = 41

avg_download_speed = 0.34

avg_upload_speed = 0.032



eie_523_class = pd.DataFrame({'Participant Video on':participant_video, 'Participant Mic On':participant_mic, 'Participant Screen Share':participant_screen, 'Others Video on':others_video, 'Others Screen Share':others_screen, 'Group':is_group, 'start_day':25, 'start_hr':start_hr, 'start_min':start_min, 'end_day':25, 'end_hr':end_hr, 'end_min':end_min, 'duration_sec':(end_min - start_min + (end_hr - start_hr)*60)*60 , 'avg_download_speed':avg_download_speed, 'avg_upload_speed':avg_upload_speed}, index=[0])

eie_523_class.head()
prediction = regressor.predict(eie_523_class)

prediction = [3 if i < 3 else i for i in prediction]

print('Total bandwidth to be consumed is: ' + str(prediction))