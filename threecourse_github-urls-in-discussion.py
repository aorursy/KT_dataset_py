import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")

data_path = "../input/"

competitions_df = pd.read_csv(data_path + "Competitions.csv")
competitions_df = competitions_df[competitions_df["CanQualifyTiers"]]
competitions_df["EnabledDate"] = pd.to_datetime(competitions_df["EnabledDate"], format="%m/%d/%Y %I:%M:%S %p")
competitions_df["DeadlineDate"] = pd.to_datetime(competitions_df["DeadlineDate"], format="%m/%d/%Y %I:%M:%S %p")
competitions_df = competitions_df.sort_values(by="DeadlineDate", ascending=False).reset_index(drop=True)
comp_tags_df = pd.read_csv(data_path + "CompetitionTags.csv")
tags_df = pd.read_csv(data_path + "Tags.csv", usecols=["Id", "Name"])

forum_messages_df = pd.read_csv(data_path + "ForumMessages.csv")
forum_topics_df = pd.read_csv(data_path + "ForumTopics.csv")
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
def github_urls_in_forum(forum_id):
    
    topic_ids = forum_topics_df[forum_topics_df["ForumId"] == forum_id]["Id"]
    _forum_messages_df = forum_messages_df[forum_messages_df["ForumTopicId"].isin(topic_ids)]
    _forum_messages_df = _forum_messages_df.merge(forum_topics_df[["Id", "Title"]].rename(columns={"Id":"ForumTopicId", "Title":"ForumTopicTitle"}))

    lst = []
    for i, r in _forum_messages_df.iterrows():
        urls = github_urls(r["Message"])
        for url in urls:
            lst.append((r["ForumTopicTitle"], url, forum_id, r["ForumTopicId"], r["Id"], r["PostDate"],))
            
    df = pd.DataFrame(lst, columns=["ForumTopicTitle", "Url", "ForumId", "ForumTopicId", "MessageId", "PostDate",])
    df = df.sort_values(["PostDate", "MessageId"])
    df = df.drop_duplicates(subset="Url", keep='first')
    
    return df

def github_urls(message):
    if message is None or pd.isnull(message):
        return []
    
    url_pattern = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
    # print(message)
    found = re.findall(url_pattern, message)
    found = [f"{t[0]}://{t[1]}{t[2]}" for t in found]
    found = [f for f in found if "github" in f]
    return found

githubs_df = []

for i, r in competitions_df.iterrows():
    # if i > 10: continue
    _df = github_urls_in_forum(r["ForumId"])
    _df.insert(0, "Slug", r["Slug"])
    _df.insert(3, "CompetitionId", r["Id"])
    _df.insert(len(_df.columns), "DeadlineDate", r["DeadlineDate"])
    
    discussion_url = "https://www.kaggle.com/c/" + _df["Slug"] + "/discussion/" + _df["ForumTopicId"].astype(str) + "#" + _df["MessageId"].astype(str)
    _df.insert(3,  "DiscussionURL", discussion_url)
    
    githubs_df.append(_df)
    # print("done..", r["Slug"])

githubs_df = pd.concat(githubs_df).reset_index(drop=True)
githubs_df.to_csv("githubs.csv")
githubs_df.head()