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
import numpy as np

import pandas as pd
data=pd.read_excel("/kaggle/input/maharashtra-engineering-cap-201920/cutoff_excel.xlsx")



pd.set_option('max_colwidth',1500)

data
cols = []

count = 1

for column in data.columns:

    cols.append(count)

    count+=1

    continue

    

data.columns = cols
data
print(data[3].value_counts(dropna=False))

print("**"*50)

print(data[5].value_counts(dropna=False))

print("**"*50)

print(data[7].value_counts(dropna=False))
for i in range(3,24,2):

    col_unwant= i

    print(col_unwant)

    data.drop(col_unwant,axis=1,inplace=True)

#data.drop('Unnamed: 2',axis=1,inplace=True)

#data.drop('Unnamed: 4',axis=1,inplace=True)

#data.drop('Unnamed: 6',axis=1,inplace=True)

#data.drop('Unnamed: 8',axis=1,inplace=True)
extra_row_01=data.index[data[1]=="II"].tolist()

data.drop(extra_row_01,inplace=True)



extra_row_02=data.index[data[1]=='Legends: Starting character G-General, L-Ladies, End character H-Home University, O-Other than Home University,S-State Level, AI- All India Seat\n. Maharashtra State Seats - Cut Off Indicates Maharashtra State General Merit No.; Figures in bracket Indicates Merit Percentile.'].tolist()

data.drop(extra_row_02,inplace=True)



extra_row_03=data.index[data[1]=='Other Than Home University Seats Allotted to Other Than Home University Candidates'].tolist()

data.drop(extra_row_03,inplace=True)



extra_row_04=data.index[data[1]=='Home University Seats Allotted to Other Than Home University Candidates'].tolist()

data.drop(extra_row_04,inplace=True)



extra_row_05=data.index[data[1]=='State Level'].tolist()

data.drop(extra_row_05,inplace=True)



extra_row_06=data.index[data[1]=='Other Than Home University Seats Allotted to Home University Candidates'].tolist()

data.drop(extra_row_06,inplace=True)



extra_row_07=data.index[data[1]=='TFWS'].tolist()

data.drop(extra_row_07,inplace=True)
cols = []

count = 1

for column in data.columns:

    cols.append(count)

    count+=1

    continue

    

data.columns = cols
data.fillna(" ",inplace=True)
idx_col_01_blanks=data.index[data[1]==" "].tolist()

idx_col_02_blanks=data.index[data[2]==" "].tolist()

idx_col_03_blanks=data.index[data[3]==" "].tolist()



l1=[]

blank_rows=[]

for i in idx_col_01_blanks:

    if i in idx_col_02_blanks:

        l1.append(i)

for i in l1:

    if i in idx_col_03_blanks:

        blank_rows.append(i)

        

data.drop(blank_rows,inplace=True)
data
data.reset_index(drop=True,inplace=True)
data_01=data.iloc[:,:2]
data_01.columns = ['text',"score"]
data_01
data_01.head()
final_dict = {}

import re



for i in range(0,data_01.shape[0]):

    try:

        m = data_01.iloc[i:i+3]

        m = m.values



        text = " ".join([str(k) for k in m[:,0]]).strip()

        values = " ".join(m[:,1]).strip()



#        print(text)



        text = text.split("\n")

        vals = values.split("\n")

    

        final_dict[i] =  {}

    

        if len(text) == 4:

            final_dict[i]['College_code'] = re.findall("[0-9]+",text[0])[0]

            final_dict[i]['College_name'] = " ".join(re.findall("[a-zA-Z,]+",text[0]))

            final_dict[i]['Subject_code'] = re.findall("[0-9]+",text[1])[0]

        

            final_dict[i]['Subject_name'] = " ".join(re.findall("[a-zA-Z,]+",text[1]))

            

            final_dict[i]['Home University'] = " ".join(re.findall("[a-zA-Z,]+",text[2].split("Home University")[1]))





            final_dict[i]['Cateogry'] = re.findall("[a-zA-Z]+",vals[0])[0]

            final_dict[i]['Rank'] = re.findall("[0-9]+",vals[0])[0]

            final_dict[i]['Score'] = re.findall("([0-9.]+)",vals[1])[0]

    

    

        elif len(text) == 3:

#             print(text)

                final_dict[i]['College_code'] = np.nan

                final_dict[i]['College_name'] = np.nan



                final_dict[i]['Subject_code'] = re.findall("[0-9]+",text[0])[0]

                final_dict[i]['Subject_name'] = " ".join(re.findall("[a-zA-Z,]+",text[0]))



                final_dict[i]['Home University'] = " ".join(re.findall("[a-zA-Z,]+",text[1].split("Home University")[1]))





                final_dict[i]['Cateogry'] = re.findall("[a-zA-Z]+",vals[0])[0]

                final_dict[i]['Rank'] = re.findall("[0-9]+",vals[0])[0]

                final_dict[i]['Score'] = re.findall("([0-9.]+)",vals[1])[0]

        else:

            pass

    except Exception as e:

        pass

data_02 = pd.DataFrame.from_records(final_dict).T.dropna(axis=0,how="all")

data_02 = data_02.drop_duplicates(['Subject_code','Subject_name'])
data_02
data_02.fillna(method="bfill").fillna(method='ffill')
data_02.reset_index(drop=True,inplace=True)
not_savitribai_clgs=data_02.index[data_02["Home University"]!="Savitribai Phule Pune University"].tolist()

print(len(not_savitribai_clgs))

data_02.drop(not_savitribai_clgs,inplace=True)
not_civil=data_02.index[data_02["Subject_name"]!="Civil Engineering"].tolist()

print(len(not_civil))

data_02.drop(not_civil,inplace=True)
data_02.reset_index(drop=True,inplace=True)
data_02
#required_cate=["GOPENH","GOPENS"]
#data_02[(data_02["Cateogry"]=="GOPENH") & (data_02["Cateogry"]=="GOPENS")]
gopenh=data_02.index[data_02["Cateogry"]=="GOPENH"].tolist()

print(gopenh)

gopens=data_02.index[data_02["Cateogry"]=="GOPENS"].tolist()

print(gopens)
not_gopens_h=[]

for i in range(0,79):

    if (i not in gopenh) & (i not in gopens):

        not_gopens_h.append(i)

print(not_gopens_h)
data_02.drop(not_gopens_h,inplace=True)
#data_02["Cateogry"]=="GOPENS"
data_02.drop(data_02.index[data_02.Rank.isnull()],inplace=True)



data_02["Rank"]=data_02["Rank"].astype(int)
data_02
data_03=data_02.sort_values(by = "Rank",ascending=True).iloc[:11,:]
data_03.reset_index(drop=True,inplace=True)
data_03
data_03.to_csv(r"Top_10_civil_clg_of_savitribai.csv",index=False)