# importing the modules needed #

import csv

from operator import sub 
# name of the target columns #

target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1',

               'ind_cco_fin_ult1','ind_cder_fin_ult1',

               'ind_cno_fin_ult1','ind_ctju_fin_ult1',

               'ind_ctma_fin_ult1','ind_ctop_fin_ult1',

               'ind_ctpp_fin_ult1','ind_deco_fin_ult1',

               'ind_deme_fin_ult1','ind_dela_fin_ult1',

               'ind_ecue_fin_ult1','ind_fond_fin_ult1',

               'ind_hip_fin_ult1','ind_plan_fin_ult1',

               'ind_pres_fin_ult1','ind_reca_fin_ult1',

               'ind_tjcr_fin_ult1','ind_valo_fin_ult1',

               'ind_viv_fin_ult1','ind_nomina_ult1',

               'ind_nom_pens_ult1','ind_recibo_ult1']



def getTarget(row):

    """

    Function to fetch the target columns as a list

    """

    tlist = []

    for col in target_cols:

        if row[col].strip() in ['', 'NA']:

            target = 0

        else:

            target = int(float(row[col]))

        tlist.append(target)

    return tlist



data_path = "../input/"

train_file = open(data_path+"train_ver2.csv")



cust_dict = {}

cust_count = 0

map_count = 0

for row in csv.DictReader(train_file):

    cust_id = int(row['ncodpers'])

    date = row['fecha_dato']

    

    if date != '2016-05-28':

        cust_dict[cust_id] = getTarget(row)  

    elif date == '2016-05-28':

        new_products = getTarget(row)

        existing_products = cust_dict.get(cust_id, [0]*24)

        num_new_products = sum([max(x1 - x2,0) for (x1, x2) in zip(new_products, existing_products)])

        if num_new_products >= 1:

            map_count += 1

        cust_count += 1

print("Number of customers in May 2016 : ",cust_count)

print("Number of customers with new products in May 2016 : ",map_count)



train_file.close()
print("Max possible MAP@7 score for May 2016: ",29712./931453.)