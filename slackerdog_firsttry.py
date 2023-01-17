target_num_map_inv = {0:'high', 1:'medium', 2:'low'}

out_df['score']= out_df['score'].apply(lambda x: target_num_map_inv[x])

dummy=pd.get_dummies(out_df['score'])
out_df=pd.concat([out_df,dummy],axis=1)
out_df["listing_id"] = test_df.listing_id.values
out_df[["listing_id","high","medium","low"]].to_csv("submission_dummy.csv",index=False)