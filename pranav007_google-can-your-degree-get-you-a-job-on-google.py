from IPython.display import Image
imgpath = ('../input/chart002/chart.png')
Image(imgpath)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
file = ("../input/google-job-skills/job_skills.csv")
df = pd.read_csv(file)
df
df.shape
## I learnt this null checking method from a kaggler (Reference - 1)(I have added his Kernal link below) 
pd.isnull(df).sum()
df = df.dropna(how ='any',axis ="rows")
pd.isnull(df).sum()
df.columns
df = df.rename(columns = {'Minimum Qualifications' : 'Min_qual','Preferred Qualifications': 'Pre_qual'})
df.columns
sns.countplot(df.Company,palette ='Set1').set_title("Company Job Counts")
y = df[["Company","Title","Category","Location","Responsibilities","Min_qual","Pre_qual"]]
yt = df[y.Company == 'YouTube']
yt
yt.shape
yt_pos = yt['Title'].apply(lambda x : x.split(",")[0])
sns.countplot(y=yt_pos,palette ='Set1').set_title("YouTube Job Title Counts")
plt.xlim(0,6)

go = df[["Company","Title","Category","Location","Responsibilities","Min_qual","Pre_qual"]]
g = df[go.Company == 'Google']
g
g.shape
## I learnt this bar chart idea from another kaggler(Reference - 2)

g_pos = g['Title'].apply(lambda x : x.split(",")[0]).value_counts()
g_pos.head(30).plot.bar()
Col_seq = ['#fc910d','#fcb13e','#239cd3','#1674b1','#ed6d50']
cat_labels = yt.Category.value_counts().sort_index().index
size = yt.Category.value_counts().sort_index()
plt.pie(size, labels = cat_labels, colors = Col_seq)
outer_c=plt.Circle( (0,0), 0.55, color='white')
inner_c=plt.Circle((0,0),0.45,color = '#fcb13e')
dot=plt.Circle((0,0),0.3,color = 'white')
p=plt.gcf().gca().add_artist(outer_c)
p1=plt.gcf().gca().add_artist(inner_c)
p2=plt.gcf().gca().add_artist(dot)
plt.figure(figsize=(10,10))
g_cat = g['Category']
sns.countplot(y=g_cat,palette = 'Set1').set_title("Google Job Category Counts")
yt_country = yt["Location"].apply(lambda x : x.split(",")[-1])
sns.countplot(y=yt_country,palette = "Set3").set_title("YouTube Job Location - Country wise")
yt_state = yt["Location"].apply(lambda x : x.split(",")[0])
yt_st = yt_state[yt_state !='Singapore']
yt_stt = yt_st[yt_st != 'London']
sns.countplot(y=yt_stt,palette = "Set2").set_title("YouTube Job - City wise")
g_country = g['Location'].apply(lambda x : x.split(',')[-1])
sns.countplot(x=g_country, palette = "Set1")
plt.xticks(rotation='vertical',fontsize=7.4)
plt.figure(figsize=(20,10))
g_city = g['Location'].apply(lambda x : x.split(',')[0])
g_cct = g_city[g_city != 'London']
g_ct = g_cct[g_cct != 'Singapore'].value_counts()
g_ct.head(30).plot.bar()
plt.xlabel('City')
plt.ylabel('Count')

prglan = ['Java ','Python','TypeScript','PHP','Elixir','Rust','SAAS','SQL','MATLAB','C/','C#','HTML','CSS','JavaScript','Android Studio','XCode',
          'MapReduce','MySQL','STATA','Swift','Ruby', 'Perl','Visual Basic','XML','SAS','Ruby','Kotlin','Objective-c','Perl']

## Learnt this method from another kaggler (Reference - 3)

yt_languages_min = dict((x,0) for x in prglan)
for i in prglan:
    x = yt['Min_qual'].str.contains(i).sum()
    if i in prglan:
        yt_languages_min[i] = x
yt_min = pd.DataFrame(list(yt_languages_min.items()),columns = ["Software",'Count'])
yt_min['Count'] = yt_min.Count.astype('int')
yt_min_count = yt_min[yt_min.Count != 0]
yt_min_count.plot.barh(y="Count", x='Software',legend = False, xlim = (0,10))
plt.title("YouTube-Popular Softwares used in Minimum Qualifications", fontsize =10)
plt.xlabel("Popularity Count")

yt_languages_pre = dict((x,0) for x in prglan)
for i in prglan:
    x = yt['Pre_qual'].str.contains(i).sum()
    if i in prglan:
        yt_languages_pre[i] = x
yt_pre = pd.DataFrame(list(yt_languages_pre.items()),columns=['Software',"Count"])
yt_pre['Count'] = yt_pre.Count.astype('int')
yt_pre_count = yt_pre[yt_pre.Count != 0]
yt_pre_count.plot.barh(y="Count", x='Software',legend = False, xlim = (0,10))
plt.title("YouTube-Popular Softwares used in Preferred Qualifications", fontsize =10)
plt.xlabel("Popularity Count")
g_languages_min = dict((x,0) for x in prglan)
for i in prglan:
    x = g['Min_qual'].str.contains(i).sum()
    if i in prglan:
        g_languages_min[i] = x
g_min = pd.DataFrame(list(g_languages_min.items()),columns=['Software',"Count"])
g_min['Count'] = g_min.Count.astype('int')
g_min_count = g_min[g_min.Count != 0]
g_min_count.plot.barh(y="Count", x='Software',legend = False, xlim = (0,200))
plt.title("Google - Popular Softwares used in Minimum Qualifications", fontsize =10)
plt.xlabel("Popularity Count")
g_languages_pre = dict((x,0) for x in prglan)
for i in prglan:
    x = g['Pre_qual'].str.contains(i).sum()
    if i in prglan:
        g_languages_pre[i] = x
g_pre = pd.DataFrame(list(g_languages_pre.items()),columns=['Software',"Count"])
g_pre['Count'] = g_pre.Count.astype('int')
g_pre_count = g_pre[g_pre.Count != 0]
g_pre_count.plot.barh(y="Count", x='Software',legend = False, xlim = (0,100))
plt.title("Google - Popular Softwares used in Preferred Qualifications", fontsize =10)
plt.xlabel("Popularity Count")
quali_degree = ['BA ','B.A','BA/BS','MA/MS', 'MA','M.A', 'MS','M.S','ME','M.E', 'PhD', 'Ph.D','MBA','MIT','IIT','IIM','CA','C.A','CPA','MST','JD','J.D','CIA','CISA','CFA','LLB','CIPD']

g_qd_min = dict((x,0) for x in quali_degree)
for i in quali_degree:
    x = g['Min_qual'].str.contains(i).sum()
    if i in quali_degree:
        g_qd_min[i] = x
g_qdmin = pd.DataFrame(list(g_qd_min.items()),columns=['Degree',"Count"])
g_qdmin['Count'] = g_qdmin.Count.astype('int')
g_qdmin_count = g_qdmin[g_qdmin.Count != 0]
g_qdmin_srtd = g_qdmin_count.sort_values('Degree')
g_qdmin_srtd.plot.barh(y="Count", x='Degree',legend = False, xlim = (0,900))
plt.title("Google - Popular Degrees Required in Minimum Qualifications", fontsize =10)
plt.xlabel("Popularity Count")

g_qd_pre = dict((x,0) for x in quali_degree)
for i in quali_degree:
    x = g['Pre_qual'].str.contains(i).sum()
    if i in quali_degree:
        g_qd_pre[i] = x
g_qdpre = pd.DataFrame(list(g_qd_pre.items()),columns=['Degree',"Count"])
g_qdpre['Count'] = g_qdpre.Count.astype('int')
g_qdpre_count = g_qdpre[g_qdpre.Count != 0]
g_qdpre_srtd = g_qdpre_count.sort_values('Degree')
g_qdpre_srtd.plot.barh(y="Count", x='Degree',legend = False, xlim = (0,250))
plt.title("Google - Popular Degrees Required in Preferred Qualifications", fontsize =10)
plt.xlabel("Popularity Count")
yt_qd_min = dict((x,0) for x in quali_degree)
for i in quali_degree:
    x = yt['Min_qual'].str.contains(i).sum()
    if i in quali_degree:
        yt_qd_min[i] = x
yt_qdmin = pd.DataFrame(list(yt_qd_min.items()),columns=['Degree',"Count"])
yt_qdmin['Count'] = yt_qdmin.Count.astype('int')
yt_qdmin_count = yt_qdmin[yt_qdmin.Count != 0]
yt_qdmin_srtd = yt_qdmin_count.sort_values('Degree')
yt_qdmin_srtd.plot.barh(y="Count", x='Degree',legend = False, xlim = (0,50))
plt.title("YouTube - Popular Degrees Required in Minimum Qualifications", fontsize =10)
plt.xlabel("Popularity Count")
yt_qd_pre = dict((x,0) for x in quali_degree)
for i in quali_degree:
    x = yt['Pre_qual'].str.contains(i).sum()
    if i in quali_degree:
        yt_qd_pre[i] = x
yt_qdpre = pd.DataFrame(list(yt_qd_pre.items()),columns=['Degree',"Count"])
yt_qdpre['Count'] = yt_qdpre.Count.astype('int')
yt_qdpre_count = yt_qdpre[yt_qdpre.Count != 0]
yt_qdpre_srtd = yt_qdpre_count.sort_values('Degree')
yt_qdpre_srtd.plot.barh(y="Count", x='Degree',legend = False, xlim = (0,20))
plt.title("YouTube - Popular Degrees Required in Preferred Qualifications", fontsize =10)
plt.xlabel("Popularity Count")
yrs_exp = ['Fresher', 'fresher', 'Intern', 'Internship','internship', 'intern',"first year student","final year student", "1 year", '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years','11 years','12 years','13 years','14 years','15 years','16 years']  
ovrmin = {}
for a in yrs_exp:
    x1 = df['Min_qual'].str.contains(a).sum()
    if a in yrs_exp:
        ovrmin[a] = x1
overminqf = pd.DataFrame(list(ovrmin.items()),columns=['Experience Category',"Count"])
overminqf['Count'] = overminqf.Count.astype('int')
overminqf_count = overminqf[overminqf.Count != 0]
overminqf_srtd = overminqf_count.sort_values('Experience Category')
overminqf_srtd.plot.barh(y="Count", x='Experience Category',legend = False, xlim = (0,200))
plt.title("Experience Category Required in Overall-Minimum Qualifications", fontsize =10)
plt.xlabel("Popularity Count")
ovrpre = {}
for a in yrs_exp:
    x1 = df['Pre_qual'].str.contains(a).sum()
    if a in yrs_exp:
        ovrpre[a] = x1
overpreqf = pd.DataFrame(list(ovrpre.items()),columns=['Experience Category',"Count"])
overpreqf['Count'] = overpreqf.Count.astype('int')
overpreqf_count = overpreqf[overpreqf.Count != 0]
overpreqf_srtd = overpreqf_count.sort_values('Experience Category')
overpreqf_srtd.plot.barh(y="Count", x='Experience Category',legend = False, xlim = (0,250))
plt.title("Experience Category Required in Overall-Preferred Qualifications", fontsize =10)
plt.xlabel("Popularity Count")
#For word popularity analysis using both case sensitive words

#First Letter Uppercase
D1 = ['Design','Computer Science','Engineering','Lawyer','Doctor','Data Science','Statistics','Fashion','Legal',
          'Visual Communication','Machine Learning','Human Computer interaction','Human-Computer','Cognitive Psychology','Anthropology',
          'Human Factors','Psychology','HCI/Computer Science',' Quantitative Finance ', 'Analytics','Sales','Service', 'Marketing',
           'Advertising','Consulting','Media','Human Resource','Management','Information Technology','Manufacturing','Industry','E-commerce',
         'Econometrics','Teacher',' Mathematics','Business','Account','Finance','Bigdata','Customer Support']
#lower case
d1 = ['design','computer science','engineering','lawyer','doctor','data science','statistics','fashion','legal',
          'visual communication','machine learning','human computer interaction','human-computer','cognitive psychology','anthropology',
          'human factors','psychology','hci/computer science',' quantitative finance ', 'analytics','sales','service', 'marketing',
           'advertising','consulting','media','human resource','management','information technology','manufacturing','industry','e-commerce',
         'econometrics','teacher',' mathematics','business','account','finance','bigdata','customer support']
D1d1 = D1+d1

role_min_qf = {}
for b in D1d1:
    x2 = df['Min_qual'].str.contains(b).sum()
    if b in D1d1:
        role_min_qf[b] = x2

role_minqf = pd.DataFrame(list(role_min_qf.items()),columns=['Domain',"Count"])
role_minqf['Count'] = role_minqf.Count.astype('int')
role_minqf_count = role_minqf[role_minqf.Count != 0]
role_minqf_srtd = role_minqf_count.sort_values('Domain')
role_minqf_srtd.plot.bar(y="Count", x='Domain',legend = False, xlim = (0,300))
plt.title("Domain Popularity in Overall-Minimum Qualifications", fontsize =10)
plt.ylabel("Popularity Count")
plt.xticks(rotation='vertical',fontsize=6.5)
plt.figure(figsize=(20,10))
role_pre_qf = {}
for b in D1d1:
    x2 = df['Pre_qual'].str.contains(b).sum()
    if b in D1d1:
        role_pre_qf[b] = x2

role_preqf = pd.DataFrame(list(role_pre_qf.items()),columns=['Domain',"Count"])
role_minqf['Count'] = role_preqf.Count.astype('int')
role_preqf_count = role_preqf[role_preqf.Count != 0]
role_preqf_srtd = role_preqf_count.sort_values('Domain')
role_preqf_srtd.plot.bar(y="Count", x='Domain',legend = False, xlim = (0,300))
plt.title("Domain Popularity in Overall-Preferred Qualifications", fontsize =10)
plt.ylabel("Popularity Count")
plt.xticks(rotation='vertical',fontsize=6.5)
plt.figure(figsize=(20,10))
