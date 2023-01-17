import numpy as np

import pandas as pd
import warnings

warnings.filterwarnings("ignore")



data_path = "../input/"



competitions_df = pd.read_csv(data_path + "Competitions.csv")

competitions_df = competitions_df[competitions_df["CanQualifyTiers"]]

competitions_df["EnabledDate"] = pd.to_datetime(competitions_df["EnabledDate"], format="%m/%d/%Y %H:%M:%S")

competitions_df["DeadlineDate"] = pd.to_datetime(competitions_df["DeadlineDate"], format="%m/%d/%Y %H:%M:%S")

competitions_df = competitions_df.sort_values(by="EnabledDate", ascending=False).reset_index(drop=True)

comp_tags_df = pd.read_csv(data_path + "CompetitionTags.csv")

tags_df = pd.read_csv(data_path + "Tags.csv", usecols=["Id", "Name"])
def get_comp_tags(comp_id):

    temp_df = comp_tags_df[comp_tags_df["CompetitionId"]==comp_id]

    temp_df = pd.merge(temp_df, tags_df, left_on="TagId", right_on="Id")

    return ", ".join(temp_df["Name"])



competitions_df["Tags"] = competitions_df.apply(lambda r: get_comp_tags(r["Id"]) , axis=1)
output_columns = ["Id","Slug","Title","HostSegmentTitle","ForumId","EnabledDate",

           "DeadlineDate","EvaluationAlgorithmAbbreviation","RewardType","RewardQuantity",

           "UserRankMultiplier","TotalTeams","TotalCompetitors","Tags"]
# competitions_df[output_columns].to_csv("../result/competitions.txt", sep="\t", index=False)

competitions_df[output_columns].to_csv("competitions.csv", index=False)

competitions_df[output_columns].head()