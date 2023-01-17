# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter('ignore')

warnings.filterwarnings('ignore')

import copy

import math



import sklearn

from sklearn.model_selection import train_test_split



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from sklearn.metrics import mean_squared_error, r2_score

from scipy import stats
pd.set_option('display.max_columns', 500)

eng_grad_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/eng_grad_emp_salary/training_set_label.csv" )

eng_grad_data.head()
test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/eng_grad_emp_salary/testing_set_label.csv')
# Checking % of NA On train data - column wise

# pd.set_option('display.max_rows', 500)

NA_col_train = pd.DataFrame(eng_grad_data.isna().sum(), columns = ['NA_Count'])

NA_col_train['% of NA'] = (NA_col_train.NA_Count/len(eng_grad_data))*100

NA_col_train.sort_values(by = ['% of NA'], ascending = False, na_position = 'first').head(10)
# Checking % of NA On train data - column wise

# pd.set_option('display.max_rows', 500)

NA_col_train = pd.DataFrame(test_data.isna().sum(), columns = ['NA_Count'])

NA_col_train['% of NA'] = (NA_col_train.NA_Count/len(test_data))*100

NA_col_train.sort_values(by = ['% of NA'], ascending = False, na_position = 'first').head(10)
X = eng_grad_data.drop(['Salary'], axis = 1)

y = eng_grad_data['Salary']
X[X.dtypes[(X.dtypes=="float64")|(X.dtypes=="int64")]

                        .index.values].hist(figsize=[16,16])
for i in X.columns:

    if X[i].nunique() == 1:

        print('With only 1 unique value: ', i)

    if X[i].nunique() == X.shape[0]:

        print('With all unique value: ', i)
for i in test_data.columns:

    if test_data[i].nunique() == 1:

        print('With only 1 unique value: ', i)

    if test_data[i].nunique() == test_data.shape[0]:

        print('With all unique value: ', i)
X = X.drop(['ID'], axis=1)

test_data = test_data.drop(['ID'], axis=1)
X['DOB'] =pd.to_datetime(X.DOB, format = '%Y-%m-%d') 

test_data['DOB'] = pd.to_datetime(test_data.DOB, format = '%Y-%m-%d')
X['year'] = X['DOB'].dt.year

X['month'] = X['DOB'].dt.month

X['day'] = X['DOB'].dt.day

X.head()
X = X.drop(['DOB'], axis=1)
test_data['year'] = test_data['DOB'].dt.year

test_data['month'] = test_data['DOB'].dt.month

test_data['day'] = test_data['DOB'].dt.day

test_data = test_data.drop(['DOB'], axis=1)

test_data.head()
cols = X.columns

num_cols = X._get_numeric_data().columns

cat_cols = list(set(cols) - set(num_cols))
cat_cols
for i in cat_cols:

    print(i)

    print(X[i].unique())
sp = X['Specialization'].unique()

print(sorted(sp))
comp = ['computer and communication engineering', 'computer application', 'computer engineering', 'computer networking', 'computer science & engineering', 'computer science and technology', 'computer science']

X['New_Specialization'] = ['Comp Science' if x in comp else x for x in X['Specialization']] 

test_data['New_Specialization'] = ['Comp Science' if x in comp else x for x in test_data['Specialization']] 
elec = ['electrical and power engineering', 'electrical engineering', 'electronics', 'electronics & instrumentation eng', 'electronics & telecommunications', 'embedded systems technology',

        'electronics and communication engineering', 'electronics and computer engineering', 'electronics and electrical engineering', 'electronics engineering']



X['New_Specialization'] = ['Electronics' if x in elec else x for x in X['New_Specialization']] 

test_data['New_Specialization'] = ['Electronics' if x in elec else x for x in test_data['New_Specialization']] 
instru = ['applied electronics and instrumentation', 'control and instrumentation engineering', 'electronics and instrumentation engineering', 'instrumentation and control engineering', 

          'instrumentation engineering', 'power systems and automation' ]

X['New_Specialization'] = ['Instrumentation' if x in instru else x for x in X['New_Specialization']] 

test_data['New_Specialization'] = ['Instrumentation' if x in instru else x for x in test_data['New_Specialization']] 
industrial = ['industrial & management engineering', 'industrial & production engineering', 'industrial engineering']

X['New_Specialization'] = ['Industrial' if x in industrial else x for x in X['New_Specialization']] 

test_data['New_Specialization'] = ['Industrial' if x in industrial else x for x in test_data['New_Specialization']] 
info = ['information & communication technology', 'information science', 'information science engineering', 'information technology']

X['New_Specialization'] = ['Information' if x in info else x for x in X['New_Specialization']] 

test_data['New_Specialization'] = ['Information' if x in info else x for x in test_data['New_Specialization']] 
mecha = ['mechanical & production engineering', 'mechanical and automation', 'mechanical engineering', 'mechatronics']

X['New_Specialization'] = ['Mechanical' if x in mecha else x for x in X['New_Specialization']] 

test_data['New_Specialization'] = ['Mechanical' if x in mecha else x for x in test_data['New_Specialization']] 
bio = ['biomedical engineering', 'biotechnology']

X['New_Specialization'] = ['Bio' if x in bio else x for x in X['New_Specialization']] 

test_data['New_Specialization'] = ['Bio' if x in bio else x for x in test_data['New_Specialization']] 
chem = ['chemical engineering', 'metallurgical engineering']

X['New_Specialization'] = ['Chemical' if x in chem else x for x in X['New_Specialization']] 

test_data['New_Specialization'] = ['Chemical' if x in chem else x for x in test_data['New_Specialization']] 
sp = X['New_Specialization'].unique()

print(sorted(sp))
sp_test = test_data['New_Specialization'].unique()

print(sorted(sp_test))
chem_test = ['polymer technology']

test_data['New_Specialization'] = ['Chemical' if x in chem_test else x for x in test_data['New_Specialization']]
auto = ['automobile/automotive engineering', 'internal combustion engine']

X['New_Specialization'] = ['Automotive' if x in auto else x for x in X['New_Specialization']] 

test_data['New_Specialization'] = ['Automotive' if x in auto else x for x in test_data['New_Specialization']] 
sp = X['New_Specialization'].unique()

print(sorted(sp))
sp_test = test_data['New_Specialization'].unique()

print(sorted(sp_test))
board = X['12board'].unique()

print(sorted(board))
ap_board = ['andhpradesh board of intermediate education', 'andhra board', 'andhra pradesh', 'andhra pradesh board of secondary education', 'andhra pradesh state board', 'ap board', 

            'ap board for intermediate education', 'ap intermediate board', 'apbie', 'apbsc', 'apsb', 'board fo intermediate education, ap', 'board of intermediate ap',

           'board of intermediate education, andhra pradesh', 'board of intermediate education, ap', 'board of intermediate education,andhra pradesh', 'board of intermediate education,andra pradesh', 

            'board of intermediate education,ap', 'board of intermediate,ap', 'board of intermidiate education,ap', 'board of intmediate education ap', 'board of secondary school of education', 

            'intermediate board of education,andhra pradesh', 'state  board of intermediate education, andhra pradesh', 'intermediate board of andhra pardesh', 'board of intrmediate education,ap']

X['New_12board'] = ['AP_BOARD' if x in ap_board else x for x in X['12board']] 

test_data['New_12board'] = ['AP_BOARD' if x in ap_board else x for x in test_data['12board']]  
up = ['board of high school and intermediate education uttarpradesh', 'bright way college, (up board)', 'bte up', 'bteup','up-board', ' upboard',

     'u p', 'u p board', 'ua', 'uo board', 'up', 'up baord', 'up board', 'up board , allahabad', 'up board allahabad', 'up board,allahabad', 'up bord', 'up bourd', 

      'up(allahabad)', 'upbhsie', 'upboard', 'uttar pradesh', 'uttar pradesh board']

X['New_12board'] = ['UP_BOARD' if x in up else x for x in X['New_12board']] 

test_data['New_12board'] = ['UP_BOARD' if x in up else x for x in test_data['New_12board']] 
dopu = ['department of pre university education', 'department of pre-university education', 'department of pre-university education(government of karnataka)', 'department of pre-university education, bangalore',

       'dept of pre-university education', 'govt of karnataka department of pre-university education', 'karnataka pre university board', 'karnataka pre unversity board',

       'karnataka pre-university', 'karnataka pre-university board', 'karnataka pu', 'karnataka pu board', 'karnataka state pre- university board', 'karnatak pu board', 

        'p u board, karnataka', 'pre university', 'pre university board', 'pre university board of karnataka', 'pre university board, karnataka', 'pre-university', 'department of pre-university eduction', 

        'pre-university board', 'psbte', 'pseb', 'pu', 'pu board', 'pu board ,karnataka', 'pu board karnataka', 'pub', 'puboard', 'puc', 'pue', 'preuniversity board(karnataka)', 'pu  board karnataka']

X['New_12board'] = ['DOPU' if x in dopu else x for x in X['New_12board']] 

test_data['New_12board'] = ['DOPU' if x in dopu else x for x in test_data['New_12board']] 
stboard = ['state bord', 'state broad', 'state syllabus', 'stateboard', 'stateboard/tamil nadu', 'staae board', 'state', 'state board', 'state board ', 'state board (jac, ranchi)', 

           'state board - tamilnadu', 'state board - west bengal council of higher secondary education : wbchse', 'state board of technical education and training', 

           'state board of technical education harayana', 'state board of technical eduction panchkula', 'hisher seconadry examination(state board)', 'intermediate state board',

          'maharashtra satate board', 'maharashtra state board', 'maharashtra state board for hsc', 'maharashtra state(latur board)', 'rbse (state board)', 'sbte, jharkhand', 'sbtet',

          'state board of karnataka', 'state board of technical education'] 

X['New_12board'] = ['STBOARD' if x in stboard else x for x in X['New_12board']] 

test_data['New_12board'] = ['STBOARD' if x in stboard else x for x in test_data['New_12board']] 
cbse = ['aissce', 'all india board', 'cbsc', 'cbse', 'cbse board', 'central board of secondary education', 'central board of secondary education, new delhi', 'certificate for higher secondary education (chse)orissa', 

        'council for indian school certificate examination', 'jawahar higher secondary school', 'cbese', 'cgbse raipur', 'aligarh muslim university']

X['New_12board'] = ['CBSE' if x in cbse else x for x in X['New_12board']] 

test_data['New_12board'] = ['CBSE' if x in cbse else x for x in test_data['New_12board']] 
wb = ['wbbhse', 'wbchse', 'wbscte', 'west bengal board of higher secondary education', 'west bengal council of higher secondary education',

     'west bengal council of higher secondary eucation', 'west bengal council of higher secondary examination (wbchse)']

X['New_12board'] = ['WB' if x in wb else x for x in X['New_12board']] 

test_data['New_12board'] = ['WB' if x in wb else x for x in test_data['New_12board']] 
bihar = ['bciec,patna', 'bice', 'bie', 'bieap', 'biec', 'biec patna', 'biec, patna', 'biec,patna', 'biec-patna', 'bihar board', 'bihar intermediate education council', 'bihar school examination board patna',

        'board of intermediate(bie)', 'bseb', 'bseb, patna', 'intermediate council patna', 'bihar', 'bihar intermediate education council, patna']

X['New_12board'] = ['Biharboard' if x in bihar else x for x in X['New_12board']] 

test_data['New_12board'] = ['Biharboard' if x in bihar else x for x in test_data['New_12board']] 
dotech = ['board of technical education', 'board of technicaleducation ,delhi', 'department of technical education', 'department of technical education, bangalore', 'haryana state board of technical education chandigarh',

         'diploma ( maharashtra state board of technical education)', 'diploma in computers', 'diploma in engg (e &tc) tilak maharashtra vidayapeeth', 'diploma(msbte)', 'directorate of technical education,banglore', 

          'dote (diploma - computer engg)', 'dpue', 'dte', 'electonincs and communication(dote)', 'punjab state board of technical education & industrial training','s j polytechnic',

         'punjab state board of technical education & industrial training, chandigarh', 'tamil nadu polytechnic', 'msbte', 'msbte (diploma in computer technology)','technical board, punchkula',

         'state board of technical education and training', 'state board of technical education harayana', 'state board of technical eduction panchkula', 'west bengal state council of technical education',

         'scte & vt (diploma)', 'scte and vt ,orissa', 'scte vt orissa', 'scte&vt', 'ks rangasamy institute of technology', 'gseb/technical education board', 'government polytechnic mumbai , mumbai board']

X['New_12board'] = ['DOTECH' if x in dotech else x for x in X['New_12board']] 

test_data['New_12board'] = ['DOTECH' if x in dotech else x for x in test_data['New_12board']] 
mp = ['mp', 'mp board', 'mp board bhopal', 'mpboard', 'mpbse', 'bsemp', 'madhya pradesh board', 'madhya pradesh open school', 'mp-bse']

X['New_12board'] = ['MP' if x in mp else x for x in X['New_12board']] 

test_data['New_12board'] = ['MP' if x in mp else x for x in test_data['New_12board']]
raj = ['board of secondary education, rajasthan', 'secnior secondary education board of rajasthan', 'secondary board of rajasthan',

      'rajasthan board', 'rajasthan board ajmer', 'rajasthan board of secondary education', 'rbse', 'board of secondary education rajasthan', 'board of secondary education,rajasthan(rbse)']

X['New_12board'] = ['RAJ' if x in raj else x for x in X['New_12board']] 

test_data['New_12board'] = ['RAJ' if x in raj else x for x in test_data['New_12board']]
isce = ['icse', 'isc', 'isc board', 'isc board , new delhi', 'isce', 'cicse']

X['New_12board'] = ['ISC' if x in isce else x for x in X['New_12board']] 

test_data['New_12board'] = ['ISC' if x in isce else x for x in test_data['New_12board']] 
hsc = ['amravati divisional board', 'aurangabad board', 'board of higher secondary examination, kerala', 'board of higher secondary orissa', 'ghseb','jaswant modern school',

      'sda matric higher secondary school', 'ssc', 'higher secondary state certificate', 'stjoseph of cluny matrhrsecschool,neyveli,cuddalore district','jaycee matriculation school',

      'hsc', 'hsc maharashtra board', 'hsc pune', 'hse', 'ahsec', 'board of intermidiate examination',  'board of secondary education', 'boardofintermediate', 'borad of intermediate',

      'higher secondary', 'higher secondary education']



X['New_12board'] = ['HSC' if x in hsc else x for x in X['New_12board']] 

test_data['New_12board'] = ['HSC' if x in hsc else x for x in test_data['New_12board']] 
other = [' board of intermediate', '0', 'baord of intermediate education', 'board of intermediate', 'board of intermediate education', 'board of intermidiate', 'ibe',

        'intermediate', 'intermediate board', 'intermediate board examination', 'intermedite', 'intermideate', 'intermidiate', 'board of intermeadiate education',

        'intermediate board of education', 'ipe', 'science college', 'sjrcw', 'matriculation', 'sri chaitanya junior kalasala', 'mpc', 'nios',

        'jiec', 'jyoti nivas', 'matric board', 'narayana junior college', 'st joseph hr sec school', 'stmiras college for girls','lucknow public college',

        'holy cross matriculation hr sec school', 'sri sankara vidyalaya', 'ssm srsecschool', 'dav public school sec 14', 'chsc']

X['New_12board'] = ['Other' if x in other else x for x in X['New_12board']] 

test_data['New_12board'] = ['Other' if x in other else x for x in test_data['New_12board']] 
ib = ['international baccalaureate (ib) diploma']

X['New_12board'] = ['IB' if x in ib else x for x in X['New_12board']] 

test_data['New_12board'] = ['IB' if x in ib else x for x in test_data['New_12board']] 
hp = ['himachal pradesh board', 'himachal pradesh board of school education']

X['New_12board'] = ['HP' if x in hp else x for x in X['New_12board']] 

test_data['New_12board'] = ['HP' if x in hp else x for x in test_data['New_12board']] 
delhi = ['dav public school', 'dav public school,hehal', 'bte,delhi', 'cbse,new delhi']

X['New_12board'] = ['DEL' if x in delhi else x for x in X['New_12board']] 

test_data['New_12board'] = ['DEL' if x in delhi else x for x in test_data['New_12board']] 
ker = ['kea', 'kerala state board', 'kerala state hse board', 'kerala', 'kerala university']

X['New_12board'] = ['KER' if x in ker else x for x in X['New_12board']] 

test_data['New_12board'] = ['KER' if x in ker else x for x in test_data['New_12board']] 
maha = ['maharashtra', 'maharashtra board', 'maharashtra board, pune', 'maharashtra nasik board', 'msbshse,pune','maharashtra state board of secondary & higher secondary education',

       'nashik board', 'pune board', 'nagpur divisional board', 'nagpur', 'nagpur board', 'kolhapur', 'latur board', 'kolhapur divisional board, maharashtra', 'latur',

       'maharashtra state board mumbai divisional board', 'nasik', 'maharashtra state boar of secondary and higher secondary education', 'ms board', 'msbte pune', 'nagpur board,nagpur']

X['New_12board'] = ['MH' if x in maha else x for x in X['New_12board']] 

test_data['New_12board'] = ['MH' if x in maha else x for x in test_data['New_12board']] 
utt = ['uttarakhand board', 'uttaranchal shiksha avam pariksha parishad', 'uttaranchal state board', 'uttrakhand board', 'uttranchal board', 'board of school education uttarakhand']

X['New_12board'] = ['UTT' if x in utt else x for x in X['New_12board']] 

test_data['New_12board'] = ['UTT' if x in utt else x for x in test_data['New_12board']] 
jhar = ['jharkhand academic council', 'jharkhand acamedic council (ranchi)', 'jharkhand acedemic council', 'jstb,jharkhand', 'jharkhand accademic council', 'jharkhand board']

X['New_12board'] = ['JHAR' if x in jhar else x for x in X['New_12board']] 

test_data['New_12board'] = ['JHAR' if x in jhar else x for x in test_data['New_12board']] 
jnk = ['j & k board', 'j&k state board of school education', 'jkbose']

X['New_12board'] = ['JNK' if x in jnk else x for x in X['New_12board']] 

test_data['New_12board'] = ['JNK' if x in jnk else x for x in test_data['New_12board']] 
guj = ['gseb', 'gsheb', 'gshseb', 'gujarat board']

X['New_12board'] = ['GUJ' if x in guj else x for x in X['New_12board']] 

test_data['New_12board'] = ['GUJ' if x in guj else x for x in test_data['New_12board']] 
tn = ['board of intermediate education,hyderabad', 'board of intermediate education:ap,hyderabad']

X['New_12board'] = ['TELAN' if x in tn else x for x in X['New_12board']] 

test_data['New_12board'] = ['TELAN' if x in tn else x for x in test_data['New_12board']] 
orissa = ['chse', 'chse, odisha', 'chse,odisha', 'chse,orissa', 'chse(concil of higher secondary education)']



X['New_12board'] = ['ORIS' if x in orissa else x for x in X['New_12board']] 

test_data['New_12board'] = ['ORIS' if x in orissa else x for x in test_data['New_12board']] 
tnadu = ['sri kannika parameswari highier secondary school, udumalpet', 'tamilnadu higher secondary education board', 'tamilnadu state board','hslc (tamil nadu state board)', 

         'tamilnadu stateboard', 'tn state board', 'srv girls higher sec school,rasipuram', 'tamil nadu state', 'tamil nadu state board']

X['New_12board'] = ['TNADU' if x in tnadu else x for x in X['New_12board']] 

test_data['New_12board'] = ['TNADU' if x in tnadu else x for x in test_data['New_12board']] 
hr = ['board of school education harayana', 'hbsc', 'hbse' ]

X['New_12board'] = ['Haryana' if x in hr else x for x in X['New_12board']] 

test_data['New_12board'] = ['Haryana' if x in hr else x for x in test_data['New_12board']] 
kn = ['karnataka board', 'karnataka board of university', 'karnataka board secondary education', 'karnataka education board', 'karnataka secondary education', 

      'karnataka secondary education board', 'karnataka sslc', 'karnataka state', 'karnataka state board', 'karanataka secondary board', 'karnataka state examination board']

X['New_12board'] = ['KN' if x in kn else x for x in X['New_12board']] 

test_data['New_12board'] = ['KN' if x in kn else x for x in test_data['New_12board']] 
board = X['New_12board'].unique()

print(sorted(board))
board_test = test_data['New_12board'].unique()

print(sorted(board_test))
board_10 = X['10board'].unique()

print(sorted(board_10))
ap = ['andhra pradesh board ssc', 'andhra pradesh state board', 'ap state board', 'ap state board for secondary education', 'apsche', 'apssc','kiran english medium high school','state board of secondary education,andhra pradesh',

     'board of secondary education - andhra pradesh', 'board of ssc education andhra pradesh', 'board ofsecondary education,ap','board of ssc education andhra pradesh', 'board ofsecondary education,ap',

     'board of secondary education,andhara pradesh', 'board of secondary education,andhra pradesh', 'board of secondary education,ap','ssc board of andrapradesh', 'ssc-andhra pradesh',

     'state board of secondary education, andhra pradesh', 'state board of secondary education, ap','school secondary education, andhra pradesh', 'board of secondary education, andhra pradesh']

X['New_10board'] = ['AP_BOARD' if x in ap else x for x in X['10board']] 

test_data['New_10board'] = ['AP_BOARD' if x in ap else x for x in test_data['10board']] 
ib = ['certificate of middle years program of ib'] 

X['New_10board'] = ['IB' if x in ib else x for x in X['New_10board']] 

test_data['New_10board'] = ['IB' if x in ib else x for x in test_data['New_10board']] 
hsc = ['hsc', 'hsce', 'hse', 'hse,board'] 

X['New_10board'] = ['HSC' if x in hsc else x for x in X['New_10board']] 

test_data['New_10board'] = ['HSC' if x in hsc else x for x in test_data['New_10board']] 
pjb = ['pseb', 'punjab school education board, mohali'] 

X['New_10board'] = ['PUN' if x in pjb else x for x in X['New_10board']] 

test_data['New_10board'] = ['PUN' if x in pjb else x for x in test_data['New_10board']] 
har = ['board of school education harayana', 'haryana board of school education', 'board of school education haryana',  'haryana board of school education,(hbse)'] 

X['New_10board'] = ['Haryana' if x in har else x for x in X['New_10board']] 

test_data['New_10board'] = ['Haryana' if x in har else x for x in test_data['New_10board']] 
jnk= ['j & k bord', 'j&k state board of school education', 'jkbose'] 

X['New_10board'] = ['JNK' if x in jnk else x for x in X['New_10board']] 

test_data['New_10board'] = ['JNK' if x in jnk else x for x in test_data['New_10board']] 
jhar = ['jbse,jharkhand', 'jharkhand academic council', 'jharkhand acedemic council', 'jharkhand secondary board', 'jharkhand secondary education board', 

        'jharkhand secondary examination board (ranchi)', 'jseb', 'jharkhand accademic council', 'jharkhand secondary examination board,ranchi'] 

X['New_10board'] = ['JHAR' if x in jhar else x for x in X['New_10board']] 

test_data['New_10board'] = ['JHAR' if x in jhar else x for x in test_data['New_10board']] 
guj = ['ghseb', 'gseb', 'gsheb', 'gujarat board', 'gujarat state board'] 

X['New_10board'] = ['GUJ' if x in guj else x for x in X['New_10board']] 

test_data['New_10board'] = ['GUJ' if x in guj else x for x in test_data['New_10board']] 
ker = ['education board of kerala', 'kea', 'kerala state board', 'kerala state technical education', 'kerala', 'kerala university'] 

X['New_10board'] = ['KER' if x in ker else x for x in X['New_10board']] 

test_data['New_10board'] = ['KER' if x in ker else x for x in test_data['New_10board']] 
icse = ['council for indian school certificate examination', 'icse board', 'icse', 'cicse'] 

X['New_10board'] = ['ICSE' if x in icse else x for x in X['New_10board']] 

test_data['New_10board'] = ['ICSE' if x in icse else x for x in test_data['New_10board']] 
raj = ['rajasthan board', 'rajasthan board ajmer', 'rajasthan board of secondary education', 'rbse', 'rbse (state board)', 'secondary board of rajasthan', 'secondary education board of rajasthan',

      'board of secondary education, rajasthan', 'board of secondary education rajasthan', 'board of secondary education,rajasthan(rbse)'] 

X['New_10board'] = ['RAJ' if x in raj else x for x in X['New_10board']] 

test_data['New_10board'] = ['RAJ' if x in raj else x for x in test_data['New_10board']] 
oris = ['board of secendary education orissa', 'board of secondary education (bse) orissa', 'board of secondary education orissa', 'board of secondary education(bse) orissa',

       'bse, odisha', 'bse,odisha', 'bse,orissa', 'hse,orissa', 'board of secondary education,odisha', 'board of secondary education,orissa', 'bsc,orissa'] 

X['New_10board'] = ['ORIS' if x in oris else x for x in X['New_10board']] 

test_data['New_10board'] = ['ORIS' if x in oris else x for x in test_data['New_10board']] 
mp = ['bsemp','madhya pradesh board','mp', 'mp board', 'mp board bhopal', 'mpboard', 'mpbse', 'mp state board', 'mp-bse'] 

X['New_10board'] = ['MP' if x in mp else x for x in X['New_10board']] 

test_data['New_10board'] = ['MP' if x in mp else x for x in test_data['New_10board']] 
tn = ['stjoseph of cluny matrhrsecschool,neyveli,cuddalore district', 'stjosephs girls higher sec school,dindigul', 'tamilnadu matriculation board', 'tamilnadu state board', 'tn state board',

     'sri kannika parameswari highier secondary school, udumalpet', 'kalaimagal matriculation higher secondary school', 'little jacky matric higher secondary school', 'tamil nadu state'] 

X['New_10board'] = ['TN' if x in tn else x for x in X['New_10board']] 

test_data['New_10board'] = ['TN' if x in tn else x for x in test_data['New_10board']] 
delhi = ['gyan bharati school', 'dav public school,hehal', 'delhi board', 'delhi public school', 'icse board , new delhi']

X['New_10board'] = ['DEL' if x in delhi else x for x in X['New_10board']] 

test_data['New_10board'] = ['DEL' if x in delhi else x for x in test_data['New_10board']] 
other = ['0','stmary higher secondary', 'mirza ahmed ali baig', 'bharathi matriculation school', 'sarada high scchool', 'anglo indian', 'don bosco maatriculation school',

         'holy cross matriculation hr sec school', "stmary's convent inter college", 'cluny', 'dav public school sec 14']

X['New_10board'] = ['Other' if x in other else x for x in X['New_10board']] 

test_data['New_10board'] = ['Other' if x in other else x for x in test_data['New_10board']] 
kar = ['karnataka', 'karnataka board of higher education', 'karnataka board of secondary education', 'karnataka education board (keeb)', 'karnataka secondary education examination board', 

       'karnataka secondary eduction', 'karnataka secondary school of examination', 'ksbe', 'kseb', 'kseeb', 'kseeb(karnataka secondary education examination board)', 'ksseb', 'ksseb(karnataka state board)',

       'karnataka secondory education board', 'karnataka sslc board bangalore', 'karnataka state education examination board', 'karnataka state secondary education board',

      'karantaka secondary education and examination borad', 'karnataka secondary board', 'karnataka state examination board', 'state board of karnataka', 'state(karnataka board)'] 

X['New_10board'] = ['KAR' if x in kar else x for x in X['New_10board']] 

test_data['New_10board'] = ['KAR' if x in kar else x for x in test_data['New_10board']] 
hp = ['hbsc', 'hbse', 'himachal pradesh board', 'himachal pradesh board of school education'] 

X['New_10board'] = ['HP' if x in hp else x for x in X['New_10board']] 

test_data['New_10board'] = ['HP' if x in hp else x for x in test_data['New_10board']] 
wb = ['wbbsce', 'wbbse', 'west bengal  board of secondary education', 'west bengal board of secondary education', 'state board - west bengal board of secondary education : wbbse',

     'west bengal board of secondary eucation', 'west bengal board of secondary examination (wbbse)'] 

X['New_10board'] = ['WB' if x in wb else x for x in X['New_10board']] 

test_data['New_10board'] = ['WB' if x in wb else x for x in test_data['New_10board']] 
utt = ['ua', 'uttarakhand board', 'uttaranchal shiksha avam pariksha parishad', 'uttaranchal state board', 'uttrakhand board', 'uttranchal board', 'board of school education uttarakhand'] 

X['New_10board'] = ['UTT' if x in utt else x for x in X['New_10board']] 

test_data['New_10board'] = ['UTT' if x in utt else x for x in test_data['New_10board']] 
ssc = ['ssc regular', 'sslc', 'sslc board', 'board of  secondary education', 'board of secondary education', 'board of secondary school education', 'board secondary  education', 'bse',

      'secondary school certificate', 'secondary school education', 'secondary school of education', 'secondary state certificate', 'board of intermediate education', 'board of secondaray education', 

       'board of ssc','bse(board of secondary education)', 'maticulation', 'matric', 'matric board', 'matriculation', 'matriculation board', 'metric', 'secondary school cerfificate', 'ssc'] 

X['New_10board'] = ['SSC' if x in ssc else x for x in X['New_10board']] 

test_data['New_10board'] = ['SSC' if x in ssc else x for x in test_data['New_10board']] 
stboard = ['state boardmp board ', 'state borad hp', 'state bord', 'stateboard','state', 'state board', 'state board ', 'state board (jac, ranchi)', 'state boardmp board ',

          'state bord', 'stateboard'] 

X['New_10board'] = ['STBOARD' if x in stboard else x for x in X['New_10board']] 

test_data['New_10board'] = ['STBOARD' if x in stboard else x for x in test_data['New_10board']] 
kn = ['karnataka board', 'karnataka board of university', 'karnataka board secondary education', 'karnataka education board', 'karnataka secondary education', 

      'karnataka secondary education board', 'karnataka sslc', 'karnataka state', 'karnataka state board', 'sslc,karnataka']

X['New_10board'] = ['KN' if x in kn else x for x in X['New_10board']] 

test_data['New_10board'] = ['KN' if x in kn else x for x in test_data['New_10board']] 
bihar = ['bihar board', 'bihar school examination board', 'bihar school examination board patna', 'bihar secondary education board,patna', 'biharboard',

        'bseb', 'bseb patna', 'bseb, patna', 'bseb,patna', 'bsepatna', 'bihar', 'bihar examination board, patna', 'bseb ,patna'] 

X['New_10board'] = ['Bihar' if x in bihar else x for x in X['New_10board']] 

test_data['New_10board'] = ['Bihar' if x in bihar else x for x in test_data['New_10board']] 
mh = ['maharashtra', 'maharashtra board', 'maharashtra board, pune', 'maharashtra nasik board', 'maharashtra satate board', 'maharashtra sate board', 'maharashtra state board', 

      'maharashtra state board for ssc', 'maharashtra state board of secondary and higher secondary education', 'maharashtra state board of secondary and higher secondary education,pune', 

      'maharashtra state board,pune', 'maharashtra state(latur board)', 'maharastra board','mhsbse', 'msbshse,pune', 'nagpur', 'nagpur board', 'nagpur divisional board', 'nashik board',

     'kolhapur', 'latur board', 'ssc maharashtra board', 'pune', 'pune board', 'sss pune', 'aurangabad board', 'nagpur board,nagpur', 'nasik', 'rbse,ajmer','maharashtra state board pune',

     'maharashtra state boar of secondary and higher secondary education', 'maharashtra state board mumbai divisional board', 'maharashtra state board of secondary & higher secondary education',

     'kolhapur divisional board, maharashtra', 'latur', 'ms board', 'mumbai board'] 

X['New_10board'] = ['MH' if x in mh else x for x in X['New_10board']] 

test_data['New_10board'] = ['MH' if x in mh else x for x in test_data['New_10board']] 
cbse = ['cbsc', 'cbse', 'cbse ', 'cbse board', 'central board of secondary education', 'central board of secondary education, new delhi', 'jawahar navodaya vidyalaya', 'national public school', 'aisse',

       'state board of secondary education( ssc)', 'cbse[gulf zone]', 'cgbse raipur'] 

X['New_10board'] = ['CBSE' if x in cbse else x for x in X['New_10board']] 

test_data['New_10board'] = ['CBSE' if x in cbse else x for x in test_data['New_10board']] 
up = ['u p', 'u p board', 'up', 'up baord', 'up board', 'up board , allahabad', 'up board allahabad', 'up board,allahabad', 'up bord', 'up borad', 'up-board',

      'up bourd', 'up(allahabad)', 'upbhsie', 'upboard', 'uttar pradesh', 'uttar pradesh board', 'bright way college, (up board)',

     'board of high school and intermediate education uttarpradesh']

X['New_10board'] = ['UP_BOARD' if x in up else x for x in X['New_10board']] 

test_data['New_10board'] = ['UP_BOARD' if x in up else x for x in test_data['New_10board']] 
assam = ['seba', 'seba(assam)']

X['New_10board'] = ['ASSAM' if x in assam else x for x in X['New_10board']] 

test_data['New_10board'] = ['ASSAM' if x in assam else x for x in test_data['New_10board']] 
board_10 = X['New_10board'].unique()

print(sorted(board_10))
board_10_test = test_data['New_10board'].unique()

print(sorted(board_10_test))
print(X.shape)

print(test_data.shape)
X = X.drop(['10board', '12board', 'Specialization', 'CollegeCityID'], axis=1)

test_data = test_data.drop(['10board', '12board', 'Specialization', 'CollegeCityID'], axis=1)

print(X.shape)

print(test_data.shape)
X.head()
X['College_by_State'] = X['CollegeID'].groupby(X['CollegeState']).transform('count')

X.head()
test_data['College_by_State'] = test_data['CollegeID'].groupby(test_data['CollegeState']).transform('count')

test_data.head()
X = X.drop(['CollegeID'], axis=1)

test_data = test_data.drop(['CollegeID'], axis=1)

print(X.shape)

print(test_data.shape)
convert_to_cat = ['Gender', '12graduation', 'CollegeTier', 'Degree', 'CollegeCityTier', 'CollegeState','GraduationYear', 'year', 'month', 'day', 

                  'College_by_State', 'New_Specialization', 'New_12board', 'New_10board']



X[convert_to_cat] = X[convert_to_cat].apply(lambda x: x.astype('category'), axis=0)

test_data[convert_to_cat] = test_data[convert_to_cat].apply(lambda x: x.astype('category'), axis=0)
cols = X.columns

num_cols = X._get_numeric_data().columns

cat_cols = list(set(cols) - set(num_cols))
X_num = len(X)

combined_dataset = pd.concat(objs=[X, test_data], axis=0)

combined_dataset = pd.get_dummies(combined_dataset, columns=cat_cols, drop_first=True)

X = copy.copy(combined_dataset[:X_num])

test_data = copy.copy(combined_dataset[X_num:])
print(X.shape)

print(test_data.shape)
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.3, random_state = 123)
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=50, max_depth=8)

dt.fit(X_train, y_train)
# Predict (train)

y_train_pred = dt.predict(X_train)



# Model evaluation (train)

mse = mean_squared_error(y_train, y_train_pred)

r2 = r2_score(y_train, y_train_pred)

rmse = math.sqrt(mse)

print('Train')

print('----------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred = dt.predict(X_val)



# Model evaluation (val)

mse = mean_squared_error(y_val, y_val_pred)

r2 = r2_score(y_val, y_val_pred)

rmse = math.sqrt(mse)

print('Val')

print('----------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
from sklearn.linear_model import RidgeCV

ridge_reg = RidgeCV(cv=5)

ridge_reg.fit(X_train, y_train)

print("Best alpha using built-in RidgeCV: %f" % ridge_reg.alpha_)

print("Best score using built-in RidgeCV: %f" % ridge_reg.score(X_train,y_train))

coef = pd.Series(ridge_reg.coef_, index = X_train.columns)
print("Ridge picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
# Predict (train)

y_train_pred = ridge_reg.predict(X_train)



# Model evaluation

mse = mean_squared_error(y_train, y_train_pred)

r2 = r2_score(y_train, y_train_pred)

rmse = math.sqrt(mse)



print('Train')

print('----------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred = ridge_reg.predict(X_val)



# Model evaluation (val)

mse = mean_squared_error(y_val, y_val_pred)

r2 = r2_score(y_val, y_val_pred)

rmse = math.sqrt(mse)

print('Val')

print('----------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
submission_rg = pd.DataFrame()

submission_rg['Salary'] = ridge_reg.predict(test_data)

submission_rg['Salary'] = submission_rg['Salary'].astype(int)

submission_rg.head()
submission_rg.to_csv("submission_rg.csv",index=False)
from sklearn.ensemble import RandomForestRegressor

rf0 = RandomForestRegressor()



param_grid = { 

     'n_estimators': [15,20,30,50,60,80,100,120],

     'max_features': ['auto', 'sqrt', 'log2'],

     'max_depth' : [2,3,5,8,9,10,11,12],

     'criterion' :['mse']

}
%%time

cv_rf = GridSearchCV(estimator=rf0, param_grid=param_grid, cv= 5)

cv_rf.fit(X_train, y_train)
cv_rf.best_params_
rfc1 = RandomForestRegressor(random_state=123, max_features='sqrt', n_estimators= 60, max_depth=12, criterion='mse')

rfc1.fit(X_train, y_train)
# Predict (train)

y_train_pred_rf = rfc1.predict(X_train)



# Model evaluation (train)

mse = mean_squared_error(y_train, y_train_pred_rf)

rmse = math.sqrt(mse)

r2 = r2_score(y_train, y_train_pred_rf)

print('Train')

print('--------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred_rf = rfc1.predict(X_val)



# Model evaluation (val)

mse = mean_squared_error(y_val, y_val_pred_rf)

rmse = math.sqrt(mse)

r2 = r2_score(y_val, y_val_pred_rf)



print('Val')

print('-------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
submission_rfgs = pd.DataFrame()

submission_rfgs['Salary'] = rfc1.predict(test_data)

submission_rfgs['Salary'] = submission_rfgs['Salary'].astype(int)

submission_rfgs.head()
submission_rfgs.to_csv("submission_rfgs.csv",index=False)
import xgboost as xgb

xgb1 = xgb.XGBRegressor()



parameters = {'learning_rate': [0.055, 0.06, 0.065],

              'max_depth': [3, 4, 5, 6,7,8],

              'min_child_weight': [4,5,6],

              'subsample':[i/10.0 for i in range(5,8)],

              'colsample_bytree': [i/10.0 for i in range(5,8)],

              'n_estimators': [60,70,75],

              'gamma':[i/10.0 for i in range(3,6)]}
xgb_gscv = GridSearchCV(xgb1, parameters, cv = 5, n_jobs = -1, verbose=True)

xgb_gscv.fit(X_train, y_train)
xgb_gscv.best_params_
# Predict (train)

y_train_pred_xgb = xgb_gscv.predict(X_train)



# Model evaluation (train)

mse = mean_squared_error(y_train, y_train_pred_xgb)

rmse = math.sqrt(mse)

r2 = r2_score(y_train, y_train_pred_xgb)



print('Train')

print('------------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred_xgb = xgb_gscv.predict(X_val)



# Model evaluation (val)

mse = mean_squared_error(y_val, y_val_pred_xgb)

rmse = math.sqrt(mse)

r2 = r2_score(y_val, y_val_pred_xgb)



print('Val')

print('-----------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
submission_xgb = pd.DataFrame()

submission_xgb['Salary'] = xgb_gscv.predict(test_data)

submission_xgb['Salary'] = submission_xgb['Salary'].astype(int)

submission_xgb.head()
submission_xgb.to_csv("submission_xgb.csv",index=False)
import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV

clf = lgb.LGBMRegressor(silent=True, random_state = 301, metric='mse', n_jobs=-1)
params ={'cat_smooth' : sp_randint(1, 100), 'max_cat_threshold': sp_randint(1,50)}

fit_params={"eval_metric" : 'mse', 

            "eval_set" : [(X_train, y_train),(X_val,y_val)],

            'eval_names': ['train','valid'],

            'verbose': 200,

            'categorical_feature': 'auto'}

gs = RandomizedSearchCV( estimator=clf, param_distributions=params, scoring='neg_root_mean_squared_error',

                        cv=5, refit=True,random_state=301,verbose=True)

gs.fit(X_train, y_train, **fit_params)

print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
gs.best_params_, gs.best_score_
clf2 = lgb.LGBMRegressor(**clf.get_params())

clf2.set_params(**gs.best_params_)
params_2 = {'learning_rate': [0.02, 0.03, 0.05, 0.08, 0.09, 0.1],   

            'num_iterations': sp_randint(30,500)}

gs2 = RandomizedSearchCV( estimator=clf2, param_distributions=params_2, scoring='neg_root_mean_squared_error',

                        cv=5, refit=True,random_state=301,verbose=True)

gs2.fit(X_train, y_train, **fit_params)

print('Best score reached: {} with params: {} '.format(gs2.best_score_, gs2.best_params_))
chk1_params = {**gs.best_params_, **gs2.best_params_, 'scoring':'neg_root_mean_squared_error'}

chk1_params
lgbm_train1 = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)

lgbm_val1 = lgb.Dataset(X_val, y_val, reference = lgbm_train1)
model_lgbm_chk1 = lgb.train(chk1_params,

                lgbm_train1,

                num_boost_round=1000,

                valid_sets=[lgbm_train1, lgbm_val1],

                feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],

                categorical_feature= [150],

                verbose_eval=100)
# Predict (train)

y_train_pred_lgbmchk1 = model_lgbm_chk1.predict(X_train)



# Model evaluation (train)

mse = mean_squared_error(y_train, y_train_pred_lgbmchk1)

rmse = math.sqrt(mse)

r2 = r2_score(y_train, y_train_pred_lgbmchk1)



print('Train')

print('--------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred_lgbmchk1 = model_lgbm_chk1.predict(X_val)



# Model evaluation (val)

mse = mean_squared_error(y_val, y_val_pred_lgbmchk1)

rmse = math.sqrt(mse)

r2 = r2_score(y_val, y_val_pred_lgbmchk1)



print('Val')

print('-------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
submission_lgbmchk1 = pd.DataFrame()

submission_lgbmchk1['Salary'] = model_lgbm_chk1.predict(test_data)

submission_lgbmchk1.head()
submission_lgbmchk1.to_csv("submission_lgbmchk1.csv",index=False)
gs2.best_params_, gs2.best_score_
clf3 = lgb.LGBMRegressor(**clf2.get_params())

clf3.set_params(**gs2.best_params_)
params_3 = {'colsample_bytree': sp_uniform(loc=0.4, scale=0.6), 'num_leaves': sp_randint(50, 500), 

            'min_child_samples': sp_randint(10,200), 'min_child_weight': [1e-2, 1e-1, 1, 1e1]}
gs3 = RandomizedSearchCV( estimator=clf3, param_distributions=params_3, scoring='neg_root_mean_squared_error',

                        cv=5, refit=True,random_state=301,verbose=True)
gs3.fit(X_train, y_train, **fit_params)

print('Best score reached: {} with params: {} '.format(gs3.best_score_, gs3.best_params_))
chk2_params = {**gs.best_params_, **gs2.best_params_, **gs3.best_params_, 'scoring':'neg_root_mean_squared_error'}

chk2_params
lgbm_train2 = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)

lgbm_val2 = lgb.Dataset(X_val, y_val, reference = lgbm_train2)
model_lgbm_chk2 = lgb.train(chk2_params,

                lgbm_train2,

                num_boost_round=1000,

                valid_sets=[lgbm_train2, lgbm_val2],

                feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],

                categorical_feature= [150],

                verbose_eval=100)
# Predict (train)

y_train_pred_lgbmchk2 = model_lgbm_chk2.predict(X_train)



# Model evaluation (train)

mse = mean_squared_error(y_train, y_train_pred_lgbmchk2)

rmse = math.sqrt(mse)

r2 = r2_score(y_train, y_train_pred_lgbmchk2)



print('Train')

print('--------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred_lgbmchk2 = model_lgbm_chk2.predict(X_val)



# Model evaluation (val)

mse = mean_squared_error(y_val, y_val_pred_lgbmchk2)

rmse = math.sqrt(mse)

r2 = r2_score(y_val, y_val_pred_lgbmchk2)



print('Val')

print('-------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
submission_lgbmchk2 = pd.DataFrame()

submission_lgbmchk2['Salary'] = model_lgbm_chk2.predict(test_data)

submission_lgbmchk2.head()
submission_lgbmchk2.to_csv("submission_lgbmchk2.csv",index=False)
gs3.best_params_, gs3.best_score_
clf4 = lgb.LGBMRegressor(**clf3.get_params())

clf4.set_params(**gs3.best_params_)
params_4 = {'max_bin': sp_randint(10, 800), 'max_depth': sp_randint(1, 10), 

            'min_data_in_leaf': sp_randint(50, 2500)}
gs4 = RandomizedSearchCV(estimator=clf4, param_distributions=params_4, scoring='neg_root_mean_squared_error',

                        cv=5, refit=True,random_state=333,verbose=True)
gs4.fit(X_train, y_train, **fit_params)

print('Best score reached: {} with params: {}'.format(gs4.best_score_, gs4.best_params_))
chk3_params = {**gs.best_params_, **gs2.best_params_, **gs3.best_params_, **gs4.best_params_,'scoring':'neg_root_mean_squared_error'}

chk3_params
lgbm_train3 = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)

lgbm_val3 = lgb.Dataset(X_val, y_val, reference = lgbm_train3)
model_lgbm_chk3 = lgb.train(chk3_params,

                lgbm_train3,

                num_boost_round=1000,

                valid_sets=[lgbm_train3, lgbm_val3],

                feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],

                categorical_feature= [150],

                verbose_eval=100)
# Predict (train)

y_train_pred_lgbmchk3 = model_lgbm_chk3.predict(X_train)



# Model evaluation (train)

mse = mean_squared_error(y_train, y_train_pred_lgbmchk3)

rmse = math.sqrt(mse)

r2 = r2_score(y_train, y_train_pred_lgbmchk3)



print('Train')

print('--------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred_lgbmchk3 = model_lgbm_chk3.predict(X_val)



# Model evaluation (val)

mse = mean_squared_error(y_val, y_val_pred_lgbmchk3)

rmse = math.sqrt(mse)

r2 = r2_score(y_val, y_val_pred_lgbmchk2)



print('Val')

print('-------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
submission_lgbmchk3 = pd.DataFrame()

submission_lgbmchk3['Salary'] = model_lgbm_chk3.predict(test_data)

submission_lgbmchk3.head()
## 200486.548832

submission_lgbmchk3.to_csv("submission_lgbmchk3.csv",index=False)
gs4.best_params_, gs4.best_score_
clf5 = lgb.LGBMRegressor(**clf4.get_params())

clf5.set_params(**gs4.best_params_)
params_5 = {'reg_lambda': sp_randint(1, 30), 'boosting': ['goss', 'dart']}
gs5 = RandomizedSearchCV(estimator=clf5, param_distributions=params_5, scoring='neg_root_mean_squared_error',

                        cv=5, refit=True,random_state=333,verbose=True)
gs5.fit(X_train, y_train, **fit_params)

print('Best score reached: {} with params: {}'.format(gs5.best_score_, gs5.best_params_))
chk4_params = {**gs.best_params_, **gs2.best_params_, **gs3.best_params_, **gs4.best_params_, **gs5.best_params_,'scoring':'neg_root_mean_squared_error'}

chk4_params
lgbm_train4 = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)

lgbm_val4 = lgb.Dataset(X_val, y_val, reference = lgbm_train4)
model_lgbm_chk4 = lgb.train(chk4_params,

                lgbm_train4,

                num_boost_round=1000,

                valid_sets=[lgbm_train4, lgbm_val4],

                feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],

                categorical_feature= [150],

                verbose_eval=100)
# Predict (train)

y_train_pred_lgbmchk4 = model_lgbm_chk4.predict(X_train)



# Model evaluation (train)

mse = mean_squared_error(y_train, y_train_pred_lgbmchk4)

rmse = math.sqrt(mse)

r2 = r2_score(y_train, y_train_pred_lgbmchk4)



print('Val')

print('-------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred_lgbmchk4 = model_lgbm_chk4.predict(X_val)



# Model evaluation (val)

mse = mean_squared_error(y_val, y_val_pred_lgbmchk4)

rmse = math.sqrt(mse)

r2 = r2_score(y_val, y_val_pred_lgbmchk4)



print('Val')

print('-------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
submission_lgbmchk4 = pd.DataFrame()

submission_lgbmchk4['Salary'] = model_lgbm_chk4.predict(test_data)

submission_lgbmchk4.head()
submission_lgbmchk4.to_csv("submission_lgbmchk4.csv",index=False)
gs5.best_params_, gs5.best_score_
clf6 = lgb.LGBMRegressor(**clf5.get_params())

clf6.set_params(**gs5.best_params_)
params_6 = {'bagging_fraction': [0.2, 0.4, 0.6, 0.8, 1], 'feature_fraction': [0.2, 0.4, 0.6, 0.8, 1]}
gs6 = RandomizedSearchCV(estimator=clf6, param_distributions=params_6, scoring='neg_root_mean_squared_error',

                        cv=5, refit=True,random_state=333,verbose=True)
gs6.fit(X_train, y_train, **fit_params)

print('Best score reached: {} with params: {}'.format(gs6.best_score_, gs6.best_params_))
final_params = {**gs.best_params_, **gs2.best_params_, **gs3.best_params_, **gs4.best_params_, 

                **gs5.best_params_, **gs6.best_params_,'scoring':'neg_root_mean_squared_error'}

final_params
lgbm_train5 = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)

lgbm_val5 = lgb.Dataset(X_val, y_val, reference = lgbm_train5)
model_lgbm_chk5 = lgb.train(final_params,

                lgbm_train5,

                num_boost_round=1000,

                valid_sets=[lgbm_train5, lgbm_val5],

                feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],

                categorical_feature= [150],

                verbose_eval=100)
# Predict (train)

y_train_pred_lgbmchk5 = model_lgbm_chk5.predict(X_train)



# Model evaluation (train)

mse = mean_squared_error(y_train, y_train_pred_lgbmchk5)

rmse = math.sqrt(mse)

r2 = r2_score(y_train, y_train_pred_lgbmchk5)





print('Train')

print('--------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred_lgbmchk5 = model_lgbm_chk5.predict(X_val)



# Model evaluation (val)

mse = mean_squared_error(y_val, y_val_pred_lgbmchk5)

rmse = math.sqrt(mse)

r2 = r2_score(y_val, y_val_pred_lgbmchk5)





print('Val')

print('-------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
submission_lgbmchk5 = pd.DataFrame()

submission_lgbmchk5['Salary'] = model_lgbm_chk5.predict(test_data)

submission_lgbmchk5.head()
submission_lgbmchk5.to_csv("submission_lgbmchk5.csv",index=False)
from sklearn.feature_selection import RFE #importing RFE class from sklearn library



rfe_xgb= RFE(estimator= xgb1 , step = 1) # with XGB , n_features_to_select=140



# Fit the function for ranking the features

fit = rfe_xgb.fit(X_train, y_train)



print("Num Features: %d" % fit.n_features_)

print("Selected Features: %s" % fit.support_)

print("Feature Ranking: %s" % fit.ranking_)
pd.set_option('display.max_rows', 100)

selected_rfe_features = pd.DataFrame({'Feature':list(X_train.columns),

                                      'Ranking':rfe_xgb.ranking_})

selected_rfe_features.sort_values(by='Ranking').reset_index(drop=True)
# Transforming the data

X_train_rfe = rfe_xgb.transform(X_train)

X_val_rfe = rfe_xgb.transform(X_val)

test_data_rfe = rfe_xgb.transform(test_data)

# Fitting our baseline model with the transformed data

rfe_xgb_model = xgb1.fit(X_train_rfe, y_train)
# Predict (train)

y_train_pred_rfe_xgb = rfe_xgb_model.predict(X_train_rfe)



# Model evaluation (train)

mse = mean_squared_error(y_train, y_train_pred_rfe_xgb)

rmse = math.sqrt(mse)

r2 = r2_score(y_train, y_train_pred_rfe_xgb)





print('Val')

print('-------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred_rfe_xgb = rfe_xgb_model.predict(X_val_rfe)



# Model evaluation (val)

mse = mean_squared_error(y_val, y_val_pred_rfe_xgb)

rmse = math.sqrt(mse)

r2 = r2_score(y_val, y_val_pred_rfe_xgb)





print('Val')

print('-------------------')

print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
submission_rfe_xgb = pd.DataFrame()

submission_rfe_xgb['Salary'] = rfe_xgb_model.predict(test_data_rfe)

submission_rfe_xgb.head()
submission_rfe_xgb.to_csv("submission_rfe_xgb.csv",index=False)