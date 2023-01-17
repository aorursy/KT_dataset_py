"""

Code based on BreakfastPirate Forum post

__author__ : SRK

"""

import csv

import datetime

from operator import sub

import numpy as np

import pandas as pd

import xgboost as xgb

from sklearn import preprocessing, ensemble
df           = pd.read_csv("../input/train_ver2.csv",

                           dtype={"sexo":str, "ind_nuevo":str, 

                                  "ult_fec_cli_1t":str, 

                                  "indext":str}, nrows=7e6) 



unique_ids   = pd.Series(df["ncodpers"].unique())

unique_id    = unique_ids.sample(n=1e4)

df           = df[df.ncodpers.isin(unique_id)]
import pandas as pd

from scipy.spatial.distance import cosine
#df.head(6).ix[:,2:8]

#df.iloc[:1, :1]
df.head()
df = df.loc[:, ["ncodpers"]].join(df.loc[:, 'ind_ahor_fin_ult1': 'ind_recibo_ult1'])
df = df.groupby("ncodpers").agg("sum")
df = df.dropna(axis=0)
data_ibs = pd.DataFrame(index=df.columns,columns=df.columns)
data_ibs.head()
# Lets fill in those empty spaces with cosine similarities

# Loop through the columns

for i in range(0,len(data_ibs.columns)) :

    # Loop through the columns for each column

    for j in range(0,len(data_ibs.columns)) :

      # Fill in placeholder with cosine similarities

      data_ibs.ix[i,j] = 1-cosine(df.ix[:,i],df.ix[:,j])
# Create a placeholder items for closes neighbours to an item

data_neighbours = pd.DataFrame(index=data_ibs.columns,columns=range(1,8))

 

# Loop through our similarity dataframe and fill in neighbouring item names

for i in range(0,len(data_ibs.columns)):

    data_neighbours.ix[i,:7] = data_ibs.ix[0:,i].order(ascending=False)[:7].index

 

# --- End Item Based Recommendations --- #
# --- Start User Based Recommendations --- #

 

# Helper function to get similarity scores

def getScore(history, similarities):

   return sum(history*similarities)/sum(similarities)
# Create a place holder matrix for similarities, and fill in the user name column

#data_sims = pd.DataFrame(index=df.index,columns=df.columns)

#data_sims.ix[:,:1] = data.ix[:,:1]
# Create a place holder matrix for similarities, and fill in the user name column

data_sims = pd.DataFrame(index=df.index,columns=df.columns)

data_sims.ix[:,:1] = df.ix[:,:1]
#data_sims.head()

data_sims.iloc[:1, :1]
#Loop through all rows, skip the user column, and fill with similarity scores

for i in range(0,len(data_sims.index)):

    for j in range(1,len(data_sims.columns)):

        user = data_sims.index[i]

        product = data_sims.columns[j]

 

        if df.iloc[i,j] >= 0:

            data_sims.iloc[i,j] = 0

        else:

            product_top_names = data_neighbours.ix[product][1:7]

            product_top_sims = data_ibs.ix[product].order(ascending=False)[1:7]

            user_purchases = df.ix[user,product_top_names]

            data_sims.ix[i][j] = getScore(user_purchases,product_top_sims) 
cat_cols = list(mapping_dict.keys())
target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

target_cols = target_cols[2:]
for row in df:

    sample = getTarget(row)
def getTarget(row):

	tlist = []

	for col in target_cols:

		if row[col].strip() in ['', 'NA']:

			target = 0

		else:

			target = int(float(row[col]))

		tlist.append(target)

	return tlist
def getIndex(row, col):

	val = row[col].strip()

	if val not in ['','NA']:

		ind = mapping_dict[col][val]

	else:

		ind = mapping_dict[col][-99]

	return ind
def getAge(row):

	mean_age = 40.

	min_age = 20.

	max_age = 90.

	range_age = max_age - min_age

	age = row['age'].strip()

	if age == 'NA' or age == '':

		age = mean_age

	else:

		age = float(age)

		if age < min_age:

			age = min_age

		elif age > max_age:

			age = max_age

	return round( (age - min_age) / range_age, 4) 
def getCustSeniority(row):

	min_value = 0.

	max_value = 256.

	range_value = max_value - min_value

	missing_value = 0.

	cust_seniority = row['antiguedad'].strip()

	if cust_seniority == 'NA' or cust_seniority == '':

		cust_seniority = missing_value

	else:

		cust_seniority = float(cust_seniority)

		if cust_seniority < min_value:

			cust_seniority = min_value

		elif cust_seniority > max_value:

			cust_seniority = max_value

	return round((cust_seniority-min_value) / range_value, 4)
def getRent(row):

	min_value = 0.

	max_value = 1500000.

	range_value = max_value - min_value

	missing_value = 101850.

	rent = row['renta'].strip()

	if rent == 'NA' or rent == '':

		rent = missing_value

	else:

		rent = float(rent)

		if rent < min_value:

			rent = min_value

		elif rent > max_value:

			rent = max_value

	return round((rent-min_value) / range_value, 6)
def processData(in_file_name, cust_dict):

	x_vars_list = []

	y_vars_list = []

	for row in csv.DictReader(in_file_name):

		# use only the four months as specified by breakfastpirate #

		if row['fecha_dato'] not in ['2015-05-28', '2015-06-28', '2016-05-28', '2016-06-28']:

			continue



		cust_id = int(row['ncodpers'])

		if row['fecha_dato'] in ['2015-05-28', '2016-05-28']:	

			target_list = getTarget(row)

			cust_dict[cust_id] =  target_list[:]

			continue

		x_vars = []

		for col in cat_cols:

			x_vars.append( getIndex(row, col) )

		x_vars.append( getAge(row) )

		x_vars.append( getCustSeniority(row) )

		x_vars.append( getRent(row) )

		if row['fecha_dato'] == '2016-06-28':

			prev_target_list = cust_dict.get(cust_id, [0]*22)

			x_vars_list.append(x_vars + prev_target_list)

		elif row['fecha_dato'] == '2015-06-28':

			prev_target_list = cust_dict.get(cust_id, [0]*22)

			target_list = getTarget(row)

			new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]

			if sum(new_products) > 0:

				for ind, prod in enumerate(new_products):

					if prod>0:

						assert len(prev_target_list) == 22

						x_vars_list.append(x_vars+prev_target_list)

						y_vars_list.append(ind)



	return x_vars_list, y_vars_list, cust_dict 
def runXGB(train_X, train_y, seed_val=0):

	param = {}

	param['objective'] = 'multi:softprob'

	param['eta'] = 0.05

	param['max_depth'] = 8

	param['silent'] = 1

	param['num_class'] = 22

	param['eval_metric'] = "mlogloss"

	param['min_child_weight'] = 1

	param['subsample'] = 0.7

	param['colsample_bytree'] = 0.7

	param['seed'] = seed_val

	num_rounds = 50



	plst = list(param.items())

	xgtrain = xgb.DMatrix(train_X, label=train_y)

	model = xgb.train(plst, xgtrain, num_rounds)	

	return model

if __name__ == "__main__":

	start_time = datetime.datetime.now()

	data_path = "../input/"

	train_file =  open(data_path + "train_ver2.csv")

	x_vars_list, y_vars_list, cust_dict = processData(train_file, {})

	train_X = np.array(x_vars_list)

	train_y = np.array(y_vars_list)

	print(np.unique(train_y))

	del x_vars_list, y_vars_list

	train_file.close()

	print(train_X.shape, train_y.shape)

	print(datetime.datetime.now()-start_time)

	test_file = open(data_path + "test_ver2.csv")

	x_vars_list, y_vars_list, cust_dict = processData(test_file, cust_dict)

	test_X = np.array(x_vars_list)

	del x_vars_list

	test_file.close()

	print(test_X.shape)

	print(datetime.datetime.now()-start_time)



	print("Building model..")

	model = runXGB(train_X, train_y, seed_val=0)

	del train_X, train_y

	print("Predicting..")

	xgtest = xgb.DMatrix(test_X)

	preds = model.predict(xgtest)

	del test_X, xgtest

	print(datetime.datetime.now()-start_time)



	print("Getting the top products..")

	target_cols = np.array(target_cols)

	preds = np.argsort(preds, axis=1)

	preds = np.fliplr(preds)[:,:7]

	test_id = np.array(pd.read_csv(data_path + "test_ver2.csv", usecols=['ncodpers'])['ncodpers'])

	final_preds = [" ".join(list(target_cols[pred])) for pred in preds]

	out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})

	out_df.to_csv('sub_xgb_new.csv', index=False)

	print(datetime.datetime.now()-start_time)