import numpy as np

import pandas as pd

pd.set_option("display.max_colwidth", 200) # use this to display more content for each column
train_df = pd.read_csv("../input/order_products__train.csv")
# we will use 5 orders with different amounts of products bought for demonstration

train_5 = train_df.loc[train_df["order_id"].isin([199872, 427287, 569586, 894112, 1890016])]



train_5.head(10)
# concatenate all product-ids into a single string

# thanks to https://www.kaggle.com/eoakley/start-here-simple-submission



def products_concat(series):

    out = ''

    for product in series:

        if product > 0:

            out = out + str(int(product)) + ' '

    

    if out != '':

        return out.rstrip()

    else:

        return 'None'
# this creates a DataFrame in the same format as your (local) prediction. 

train_5 = pd.DataFrame(train_5.groupby('order_id')["product_id"].apply(products_concat)).reset_index()

train_5
df_real = pd.DataFrame({"order_id": [10, 11, 12, 13, 14, 15, 16],

                        "product_id": ["0", "0 1", "0 1 2 3",  "0 1 2 3", "0 1 2 3 4 5", 

                                       "0 1 2 3 4 5 6 7", "0 1 2 3 4 5 6 7 8 9"]},

                       index=np.arange(7))



df_pred = pd.DataFrame({"order_id": [10, 11, 12, 13, 14, 15, 16], 

                        "product_id": ["0", "0 X", "0 1 2 Y", "0 1 2 3 4 5", "0 1 2 3", 

                                       "0 1 2 3 4 5 6 7 8 9 X", "0 1 2 3 4 5 6 7 8"]},

                       index=np.arange(7))



df_real_preds = pd.merge(df_real, df_pred, on="order_id", suffixes=("_real", "_pred"))

df_real_preds
def score_order_predictions(df_real, df_pred, return_df=True, show_wrong_IDs=True):

    '''

    Print out the total weighted precision, recall and F1-Score for the given true and predicted orders.

    

    return_df:  if set to True, a new DataFrame with added columns for precision, recall and F1-Score will be returned.

    

    show_wrong_IDs: if set to True, two columns with the IDs that the prediction missed and incorrectly predicted will be added. Needs return_df to be True.

    '''

    df_combined = pd.merge(df_real, df_pred, on="order_id", suffixes=("_real", "_pred"))

    

    df_combined["real_array"] = df_combined["product_id_real"].apply(lambda x: x.split())

    df_combined["pred_array"] = df_combined["product_id_pred"].apply(lambda x: x.split())

    

    df_combined["num_real"] = df_combined["product_id_real"].apply(lambda x: len(x.split()))

    df_combined["num_pred"] = df_combined["product_id_pred"].apply(lambda x: len(x.split()))



    df_combined["num_pred_correct"] = np.nan

    for i in df_combined.index:

        df_combined.loc[i, "num_pred_correct"] = len([e for e in df_combined.loc[i,"real_array"]

                                                      if e    in df_combined.loc[i,"pred_array"]])

    if show_wrong_IDs==True:

        df_combined["IDs_missing"] = np.empty((len(df_combined), 0)).tolist()

        for i in df_combined.index:

            missing = np.in1d(df_combined.loc[i, "real_array"], df_combined.loc[i,"pred_array"], invert=True)

            missing_values = np.array(df_combined.loc[i, "real_array"])[missing]

            df_combined.set_value(i, "IDs_missing", missing_values)

 

        df_combined["IDs_not_ordered"] = np.empty((len(df_combined), 0)).tolist()

        for i in df_combined.index:

            not_ordered = np.in1d(df_combined.loc[i, "pred_array"], df_combined.loc[i,"real_array"], invert=True)

            not_ordered_values = np.array(df_combined.loc[i, "pred_array"])[not_ordered]

            df_combined.set_value(i, "IDs_not_ordered", not_ordered_values)



    df_combined["precision"] = np.round(df_combined["num_pred_correct"] / df_combined["num_pred"], 4)

    df_combined["recall"]    = np.round(df_combined["num_pred_correct"] / df_combined["num_real"], 4)

    df_combined["F1-Score"]  = np.round(2*( (df_combined["precision"]*df_combined["recall"]) / 

                                           (df_combined["precision"]+df_combined["recall"]) ), 4)

    

    recall_total =    df_combined["num_pred_correct"].sum() / df_combined["num_real"].sum()

    precision_total = df_combined["num_pred_correct"].sum() / df_combined["num_pred"].sum()

    F1_total =  2* ( (precision_total * recall_total) / (precision_total + recall_total) )      

    

    print("F1-Score: ", round(F1_total, 4))

    print("recall:   ", round(recall_total, 4))

    print("precision:", round(precision_total, 4))

    

    df_combined.drop(["real_array", "pred_array", "num_real", "num_pred", "num_pred_correct"], axis=1, inplace=True)

    

    # reorder columns so that the scoring-columns appear first and

    # all other on the right of them (bad readability with many IDs)

    df_combined = pd.concat([df_combined.loc[:, "order_id"], 

                             df_combined.iloc[:, -3:],

                             df_combined.iloc[:, 1:-3]], 

                            axis=1)

    if return_df==True:

        return df_combined

    else: 

        return None
df_scores = score_order_predictions(df_real, df_pred, return_df=True, show_wrong_IDs=True)

df_scores