import scipy

import pandas as pd

from sklearn.linear_model import LogisticRegression
D0 = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv", index_col="id")

D_test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv", index_col="id")



y_train = D0["target"]

D = D0.drop(columns="target")

test_ids = D_test.index



D_all = pd.concat([D, D_test])

num_train = len(D)



print(f"D_all.shape = {D_all.shape}")        
for col in D.columns.difference(["id"]):

    train_vals = set(D[col].dropna().unique())

    test_vals = set(D_test[col].dropna().unique())



    xor_cat_vals = train_vals ^ test_vals

    if xor_cat_vals:

        print(f"Replacing {len(xor_cat_vals)} values in {col}, {xor_cat_vals}")

        D_all.loc[D_all[col].isin(xor_cat_vals), col] = "xor"
ord_maps = {

    "ord_0": {val: i for i, val in enumerate([1, 2, 3])},

    "ord_1": {

        val: i

        for i, val in enumerate(

            ["Novice", "Contributor", "Expert", "Master", "Grandmaster"]

        )

    },

    "ord_2": {

        val: i

        for i, val in enumerate(

            ["Freezing", "Cold", "Warm", "Hot", "Boiling Hot", "Lava Hot"]

        )

    },

    **{col: {val: i for i, val in enumerate(sorted(D_all[col].dropna().unique()))} for col in ["ord_3", "ord_4", "ord_5", "day", "month"]},

}
oh_cols = D_all.columns.difference(ord_maps.keys() - {"day", "month"})



print(f"OneHot encoding {len(oh_cols)} columns")



one_hot = pd.get_dummies(

    D_all[oh_cols],

    columns=oh_cols,

    drop_first=True,

    dummy_na=True,

    sparse=True,

    dtype="int8",

).sparse.to_coo()
ord_cols = pd.concat([D_all[col].map(ord_map).fillna(max(ord_map.values())//2).astype("float32") for col, ord_map in ord_maps.items()], axis=1)

ord_cols /= ord_cols.max()  # for convergence



ord_cols_sqr = 4*(ord_cols - 0.5)**2
X = scipy.sparse.hstack([one_hot, ord_cols, ord_cols_sqr]).tocsr()

print(f"X.shape = {X.shape}")



X_train = X[:num_train]

X_test = X[num_train:]
clf=LogisticRegression(C=0.05, solver="lbfgs", max_iter=5000)



clf.fit(X_train, y_train)



pred = clf.predict_proba(X_test)[:, 1]



pd.DataFrame({"id": test_ids, "target": pred}).to_csv("submission.csv", index=False)