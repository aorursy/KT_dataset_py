import pandas as pd



group_id = pd.read_csv('/kaggle/input/bangumi-group-project/group_id.csv')

topic_id = pd.read_csv('/kaggle/input/get-topics-links/topic_id.csv')

topic_content = pd.read_csv('/kaggle/input/bgm-topic-content/topic_content.csv')

subject_id = pd.read_csv('/kaggle/input/bgm-subject-go/topic_info.csv')

subject_content = pd.read_csv('/kaggle/input/bgm-subject-go/topic_content.csv')



# 没爬下来帖有两个：351669和13453，内容很无语，我就直接扔了

topic_content = topic_content.loc[(topic_content.postId != 'missed') &

# 另外，没爬下来的回复有19个，由于没有topicId来重爬，我也直接扔了。（不好意思了。。。）

                                  topic_content.dateTime.notnull() &

# 还有，有个老哥的id居然是空的，我佛了。。。他的两个回复我也扔了。参考 bgm.tv/group/topic/33305 二楼

                                  topic_content.userId.notnull()].copy()



# additional manipulation

topic_content.postId = topic_content.postId.map(int).copy()

subject_id.subjectId = subject_id.subjectId.map(lambda i: i[9:])
#topic_content.loc[topic_content.floor.str.contains('-')].sort_values('postId')
# 剩下的没什么要改的，去重一下就完成了

group_id.drop_duplicates().set_index('groupId').to_csv('group_id.csv')

topic_id.drop_duplicates().set_index('topicId').to_csv('topic_id.csv')

topic_content.sort_values('postId').set_index('postId').to_csv('topic_content.csv')

subject_id.set_index('topicId').to_csv('subject_id.csv')

subject_content.set_index('postId').to_csv('subject_content.csv')