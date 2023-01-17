import gc



import joblib

import numpy as np

import pandas as pd

from scipy import sparse

from sklearn.decomposition import NMF
Competitions = pd.read_csv("/kaggle/input/meta-kaggle/Competitions.csv")

Users = pd.read_csv("/kaggle/input/meta-kaggle/Users.csv")

Teams = pd.read_csv("/kaggle/input/meta-kaggle/Teams.csv")

TeamMemberships = pd.read_csv("/kaggle/input/meta-kaggle/TeamMemberships.csv")



df = pd.merge(

    Teams,

    Competitions[Competitions.CanQualifyTiers],

    left_on="CompetitionId",

    right_on="Id",

    how="left",

    suffixes=("_team", "_comp")

)



df = pd.merge(

    TeamMemberships,

    df,

    left_on="TeamId",

    right_on="Id_team",

    how="left",

    suffixes=("_team_mem", "")

)



df = pd.merge(

    Users.query("PerformanceTier>0"),

    df,

    left_on="Id",

    right_on="UserId",

    how="left",

    suffixes=("_user", "")

)



use_cols = ["UserName", "CompetitionId"]

df = df[use_cols].dropna()



del Competitions, Users, Teams, TeamMemberships

gc.collect()



print(df.shape)
cross = pd.crosstab(df["UserName"], df["CompetitionId"])

X = sparse.csr_matrix(cross, dtype=np.int8)

print(X.shape)
model = NMF(n_components=2, init='random', random_state=0)

W = model.fit_transform(X)

H = model.components_

R = np.dot(W,H)
W.shape, H.shape, R.shape
cross.head()
pd.DataFrame(R).head()
user_name = "sishihara"

user_index = cross.index.get_loc(user_name)

joined_competitions = list(cross.loc[user_name][(cross.loc[user_name] > 0).values].index)

candidate_competitions = list(cross.loc[user_name].index[(-1 * R[user_index]).argsort()])

recommend_competidions = [int(cc) for cc in candidate_competitions[:10] if cc not in joined_competitions]

print(f"{user_name}: {len(joined_competitions)}")
Competitions = pd.read_csv("/kaggle/input/meta-kaggle/Competitions.csv")

Competitions[Competitions["Id"].isin(recommend_competidions)]
joblib.dump(cross, "UsersCompetitionsMatrix.pkl")

joblib.dump(R, "Recommendations.pkl")
joblib.__version__