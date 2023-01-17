import numpy as np

import pandas as pd

import plotly.express as px
js_df = pd.read_csv('../input/ekosistem-javascript-di-indonesia/Ekosistem JavaScript (Responses) - Form Responses 1.csv')
js_df.head()
to_drop = ['Timestamp', 'Domisili saat ini (kota, provinsi).1', 'Unnamed: 7', 'Kamu sedang tertarik belajar tentang framework apa?', 'Bagaimana biasanya kamu belajar sebuah teknologi atau framework?', 'Bagaimana pendapatmu tentang framework tersebut? (Kelebihan dan kekurangan)', 'Saat ini sedang tertarik belajar framework apa?', 'Perusahaan kamu saat ini memiliki...', 'Saat mencari talenta, biasanya banyak merekrut dari mana?', 'Apakah tim development sudah menggunakan testing dalam proses development']



# drop the list

js_df.drop(to_drop, inplace=True, axis=1)



# replace column's name

js_df.columns = ['city', 'gender', 'age', 'js_community', 'job', 'media_to_learn_code', 'work_experience', 'framework', 'consederation_on_choosing_new_tech', 'salary', 'testing_tools']
js_df.head()
missing_values_count = js_df.isnull().sum()



# nan by columns

missing_values_count
# how many total missing values do we have?

total_cells = np.product(js_df.shape)

total_missing = missing_values_count.sum()



# percent of data that is missing

(total_missing/total_cells) * 100
js_df['city'].unique()
# cleaning city's name

js_df['city'] = np.where(js_df['city'].str.contains('Bali|bali|Denpasar'), 'Bali', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Malang|malang'), 'Malang', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Jakarta|jakarta|Jkt|jkt|JKT'), 'Jakarta', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Surabaya|surabaya|SURABAYA'), 'Surabaya', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Bandung|bandung'), 'Bandung', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('(?i)Yogyakarta|Jogjakarya'), 'Yogyakarta', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Tangerang|tangerang'), 'Tangerang Selatan', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Bogor|bogor|Cibinong'), 'Bogor', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Depok|depok'), 'Depok', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Bekasi|bekasi'), 'Bekasi', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Temanggung|temanggung'), 'Temanggung', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Karanganyar|karanganyar'), 'Karanganyar', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Taipei|taipie'), 'Taipei', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Taipei|taipie'), 'Taipei', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Cirebon|cirebon'), 'Cirebon', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Mataram|mataram'), 'Mataram', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Probolinggo|probolinggo'), 'Probolinggo', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Gresik|gresik'), 'Gresik', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('Medan'), 'Medan', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('(?i)Batang'), 'Batang', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('(?i)semarang'), 'Semarang', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('(?i)makassar'), 'Makassar', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('(?i)purwokerto'), 'Purwokerto', js_df['city'])

js_df['city'] = np.where(js_df['city'].str.contains('(?i)palu'), 'Palu', js_df['city'])
js_df['city'].unique()
city_df = pd.DataFrame( data = {'city': (js_df['city'].value_counts()).index, 'user':(js_df['city'].value_counts()).values}).sort_values('user', ascending=False)
fig = px.bar(city_df, x='city', y='user', text='user')

fig.update_layout(

    title="Total user in each city"

)

fig.show()
js_df['gender'].value_counts()
gender_df = pd.DataFrame(data = {'gender':(js_df['gender'].value_counts()).index, 'total':(js_df['gender'].value_counts()).values}).sort_values('total', ascending=False)
fig = px.bar(gender_df, x="gender", y="total", color='gender')

fig.update_layout(

    title="Gender distribution"

)

fig.show()
fig = px.pie(gender_df, values='total', names='gender')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(

    title="Gender percentage"

)

fig.show()
js_df['age'] = np.where(js_df['age'].str.contains('(?i)tahun'), '31', js_df['age'])
# make data frame

age_df = pd.DataFrame( data = {'age':(js_df['age'].value_counts()).index, 'total':(js_df['age'].value_counts()).values}).sort_values('total', ascending=False)
#cleaning age column

age_df = age_df[~age_df['age'].str.contains('berumur')]



# convert age column data types into int

age_df['age'] = age_df['age'].astype(int)



# check age_df data types

age_df.dtypes
fig = px.bar(age_df, x='age', y='total', text='total')

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(title="Age distribution", uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
fig = px.pie(age_df, values='total', names='age')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(title="Age percentage")

fig.show()
# make data frame

comm_df = pd.DataFrame(js_df.js_community.str.split(',').tolist(), index=js_df.city).stack()

comm_df = comm_df.reset_index([0, 'city'])

comm_df.columns = ['city', 'community_name']

comm_df['community_name'] = comm_df['community_name'].str.strip()
#cleaning community's name

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)bali js|balijs|www.facebook.com'), 'BaliJS', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)jakarta js|jakartajs'), 'JakartaJS', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)bandung js|bandungjs'), 'BandungJS', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)jogja js|jogjajs'), 'JogjaJS', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)surabaya js|surabayajs'), 'SurabayaJS', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)minang js|minangjs'), 'MinangJS', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)lombok js|lombokjs'), 'LombokJS', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)ReactJs Sg'), 'Singapore', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)react'), 'ReactJS ID', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)vue'), 'VueJS ID', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)javascript indonesia|js indo'), 'JavaScript Indonesia', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)DevC Jakarta|Facebook Developer Circle: Jakarta|Facebook Developer Circle Jakarta|Developer Circles Jakarta|FB|FB dev circle'), 'Facebook Developer Circle: Jakarta', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)Facebook Developer Circle: Bandung|DevC Bandung|Facebook Dev Bandung'), 'Facebook Developer Circle: Bandung', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)semarang'), 'SemarangJS', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)gresik'), 'Gresik Dev', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)node'), 'NodeJS ID', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)Developer Circle Bali|FB Developer Circle'), 'Facebook Developer Circle: Bali', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)FB Dev C'), 'Facebook Developer Circle: Malang', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)Dev C'), 'Facebook Developer Circle: Yogyakarta', comm_df.loc[:,'community_name'])

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].str.contains('(?i)telegram'), 'Telegram', comm_df.loc[:,'community_name'])
# community name row drop list

drop_list = ['Belum bergabung', 'Tidak Tahu', 'Belum', 'Tidak ada', 'Tidak',

             'belum tau', 'Mungkin ada tp tidak ikut', 'belum tau , tapi daerah depok pasti ada',

             'Tidak tahu', 'Tidak Ada', 'tidak ada', 'banyak', 'Ada', 'Banyak',

             '-', 'Tidak tahu, tapi ingin join jika ada', 'Sepertinya ada tapi saya tidak mengetahuinya',

             'Belum Tau', 'Belum ada', 'setahu saya tidak ada', 'Kurang tau', 'gk tau', 'setau saya tidak ada',

             'belum tau ', ' tapi ingin join jika ada', 'ada', 'Yang saya thu', 'tapi daerah depok pasti ada',

             'Saya tinggal di bekasi tapi ngantor jakarta', 'Ya', 'ngikut komunitas jakarta', 'tapi ingin join jika ada', 

            ]
# drop the list

comm_df.loc[:,'community_name'] = np.where(comm_df.loc[:,'community_name'].isin(drop_list), np.nan, comm_df.loc[:,'community_name'])
# make another data frame

nComm_df = pd.DataFrame(comm_df.groupby(['city', 'community_name'])['community_name'].count(), index=(comm_df.groupby(['city', 'community_name'])['community_name'].count()).index)

nComm_df = nComm_df.rename(columns={"community_name": "total"})

nComm_df = nComm_df.reset_index(level='community_name')

nComm_df = nComm_df.reset_index()
fig = px.bar(nComm_df, x="total", y="city", color='community_name', orientation='h')

fig.update_layout(title="JS community in each city")

fig.show()
# make data frame

job_df = pd.DataFrame(js_df.job.str.split(' dan ').tolist(), index=js_df.city).stack()

job_df = job_df.reset_index([0, 'city'])

job_df.columns = ['city', 'job']
#cleaning job's name

job_df.loc[:,'job'] = np.where(job_df.loc[:,'job'].str.contains('(?i)frontend'), 'Frontend', job_df.loc[:,'job'])

job_df.loc[:,'job'] = np.where(job_df.loc[:,'job'].str.contains('(?i)IT Support'), 'IT Support', job_df.loc[:,'job'])

job_df.loc[:,'job'] = np.where(job_df.loc[:,'job'].str.contains('(?i)STAFF KONSULTAN BISNIS'), 'Staff Konsultan Bisnis', job_df.loc[:,'job'])

job_df.loc[:,'job'] = np.where(job_df.loc[:,'job'].str.contains('(?i)fullstack'), 'Fullstack', job_df.loc[:,'job'])

job_df.loc[:,'job'] = np.where(job_df.loc[:,'job'].str.contains('(?i)freelance'), 'Freelance', job_df.loc[:,'job'])

job_df.loc[:,'job'] = np.where(job_df.loc[:,'job'].str.contains('(?i)it solution'), 'IT Solution', job_df.loc[:,'job'])

job_df.loc[:,'job'] = np.where(job_df.loc[:,'job'].str.contains('(?i)junior'), 'Junior Programmer', job_df.loc[:,'job'])

job_df.loc[:,'job'] = np.where(job_df.loc[:,'job'].str.contains('(?i)Sedang kuliah'), 'Sedang Kuliah', job_df.loc[:,'job'])

job_df.loc[:,'job'] = np.where(job_df.loc[:,'job'].str.contains('(?i)Pelajat'), 'Sedang Sekolah', job_df.loc[:,'job'])

job_df.loc[:,'job'] = np.where(job_df.loc[:,'job'].str.contains('(?i)Baru lulus'), 'Baru lulus kuliah', job_df.loc[:,'job'])

job_df.loc[:,'job'] = np.where(job_df.loc[:,'job'].str.contains('(?i)backend'), 'Backend', job_df.loc[:,'job'])
fig = px.pie(job_df['job'].value_counts(), names=(job_df['job'].value_counts()).index, values=(job_df['job'].value_counts()).values)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(title="Job percentage")

fig.show()
# make another data frame

njob_df = pd.DataFrame(job_df.groupby(['city', 'job'])['job'].count(), index=(job_df.groupby(['city', 'job'])['job'].count()).index)

njob_df = njob_df.rename(columns={"job": "total"})

njob_df = njob_df.reset_index(level=['city', 'job'])
fig = px.bar(njob_df, x="total", y="city", color='job', orientation='h')

fig.update_layout(title="Job in each city")

fig.show()
fig = px.pie(js_df, names=(js_df['work_experience'].value_counts()).index, values=(js_df['work_experience'].value_counts()).values)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(title="Work experience percentage")

fig.show()
fig = px.pie(js_df, names=(js_df['salary'].value_counts()).index, values=(js_df['salary'].value_counts()).values)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(title="Salary percentage")

fig.show()
# make data frame

salary_df = pd.DataFrame(js_df.groupby(['city', 'salary'])['salary'].count(), index=(js_df.groupby(['city', 'salary'])['salary'].count()).index)

salary_df = salary_df.rename(columns={"salary": "total"})

salary_df = salary_df.reset_index(level=['city', 'salary'])
fig = px.bar(salary_df, x="total", y="city", color='salary', orientation='h')

fig.update_layout(title="Salary in each city")

fig.show()
fig = px.bar(js_df, x=(js_df['framework'].value_counts()).index, y=(js_df['framework'].value_counts()).values)

fig.update_layout(title="JavaScript framework distribution")

fig.show()
fig = px.pie(js_df, names=(js_df['framework'].value_counts()).index, values=(js_df['framework'].value_counts()).values)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(title="JavaScript framework percentage")

fig.show()
tTools_df = pd.DataFrame(data = {'testing_tools':(js_df['testing_tools'].value_counts()).index, 'total':(js_df['testing_tools'].value_counts()).values}).sort_values('total', ascending=False)

tTools_df = tTools_df[~tTools_df.testing_tools.str.contains('di|belum')]

tTools_df.loc[:,'testing_tools'] = np.where(tTools_df.loc[:,'testing_tools'].str.contains('AB'), 'A/B Testing', tTools_df.loc[:,'testing_tools'])

tTools_df.loc[:,'testing_tools'] = np.where(tTools_df.loc[:,'testing_tools'].str.contains('(?i)mocha'), 'Mocha', tTools_df.loc[:,'testing_tools'])
fig = px.bar(tTools_df, x='testing_tools', y='total')

fig.update_layout(title="Testing tools distribution")

fig.show()
fig = px.pie(tTools_df, names='testing_tools', values='total')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(title="Testing tools percentage")

fig.show()