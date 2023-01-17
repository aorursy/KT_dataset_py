import matplotlib as mpl

from matplotlib import font_manager as fm

import matplotlib.pyplot as plt

from wordcloud import WordCloud

font_list = fm.findSystemFonts(fontpaths= None, fontext='ttf')

print('설정위치',mpl.matplotlib_fname())
path = '../input/nanumbarunpen/NanumBarunpenRegular.ttf'

font_name = fm.FontProperties(fname=path, size=50).get_name()

print(font_name)

plt.rc('font', family=font_name)
import pandas as pd



youtube_raw_data = pd.read_csv('../input/youtube-new/KRvideos.csv', engine='python')
youtube_raw_data
import json



category_list = []

category_id = []

for i in json.load(open('../input/youtube-new/KR_category_id.json','r'))['items']:

    category_list.append(i['snippet']['title'])

for i in json.load(open('../input/youtube-new/KR_category_id.json','r'))['items']:

    category_id.append(i['id'])
def dictionary_gen(title, iden):

    title_range = len(title)

    iden_range = len(iden)

    

    category_dict = {}

    if title_range == iden_range:

        for i in range(title_range):

            category_dict[int(iden[i])] = title[i]

    else:

        return(0)

    

    return category_dict

        
category_dict = dictionary_gen(category_list, category_id)
category_dict
category_dict_match = {}



for i in list(category_dict.keys()):

    if i in youtube_raw_data.category_id.value_counts().index:

        category_dict_match[i] = category_dict[i]
category_dict_match
for i in list(category_dict_match.keys()):

    youtube_raw_data.loc[youtube_raw_data.category_id == i, 'category'] = category_dict_match[i]
youtube_raw_data[youtube_raw_data.duplicated('title')].title
youtube_raw_data = youtube_raw_data.drop_duplicates('title')
youtube_raw_data['tags_list'] = youtube_raw_data.tags.apply(lambda x: x.split('|') )
youtube_raw_data['tags_list']
tag_list_vector = []

for i in youtube_raw_data['tags_list']:

    length = len(i)

    for j in range(length):

        tag_list_vector.append(i[j])
trend_with_tags = pd.Series(tag_list_vector)[pd.Series(tag_list_vector) != '[none]'].value_counts()

trend_with_tags = trend_with_tags[trend_with_tags.values >= 5]
tag_list_vector.remove('[none]')
trend = dict(trend_with_tags)

trend
wordcloud = WordCloud(

            font_path = path,

            width = 800,

            height = 800,

            background_color='white'

            )

wordcloud = wordcloud.generate_from_frequencies(trend)
array = wordcloud.to_array()

plt.figure(figsize=(13,10))

plt.imshow(array)
import seaborn as sns

plt.figure(figsize=(20,9))



count = sns.countplot(data=youtube_raw_data,

                      x= 'category',

                       order=list(youtube_raw_data.category.value_counts().index))



count.set_xticklabels(count.get_xticklabels(),rotation=45)

count.set_title('Amount of video', fontsize = 20)

count.set_xlabel('category', fontsize = 20)

count.set_ylabel('count', fontsize = 20)

count.tick_params(labelsize = 18)
over_million_view = youtube_raw_data[youtube_raw_data.views >= 1000000]

plt.figure(figsize=(17,9))

upper_mil_count = sns.countplot(data=over_million_view,

                                x= 'category',

                                order= over_million_view.category.value_counts().index

                                )

                                





upper_mil_count.set_xticklabels(upper_mil_count.get_xticklabels(),rotation=45)

upper_mil_count.set_xlabel('category', fontsize = 20)

upper_mil_count.set_ylabel('count', fontsize = 20)



upper_mil_count.set_title('Video category upper 1m view', fontsize = 20)

upper_mil_count.tick_params(labelsize = 18)
liked_video = youtube_raw_data.sort_values('likes',ascending=False)[:1000]

plt.figure(figsize=(17,9))

most_liked_video = sns.countplot(data= liked_video,

                                 x= 'category',

                                 order= liked_video.category.value_counts().index)

                                





most_liked_video.set_xticklabels(most_liked_video.get_xticklabels(),rotation=45)

most_liked_video.set_xlabel('category', fontsize = 20)

most_liked_video.set_ylabel('count', fontsize = 20)



most_liked_video.set_title('more liked 1000video', fontsize = 20)

most_liked_video.tick_params(labelsize = 18)


# 카운트 플롯이 아닌 숫자연산을 위하여 다른 데이터 프레임을 다시 만들었습니다.

ratio_with_all = pd.DataFrame(youtube_raw_data.category.value_counts())

#컬럼 네임 변경

ratio_with_all.columns = ['count']

#카테고리 컬럼에 index로 들어가있는 카테고리를 집어넣음

ratio_with_all['category'] = ratio_with_all.index

#좋아요 순위 1000위 안에 들어가 있는 카테고리만으로 ratio_with_all 변경

ratio_with_all = ratio_with_all.loc[list(liked_video.category.value_counts().index)]

#전체 동영상 수로 1000위 안에 들어가있는 카테고리 동영상 수를 나눈 후 *100을 하여 %값 표현

ratio_with_all['divided_by_allcount'] = (liked_video.category.value_counts()/ratio_with_all['count']) * 100
plt.figure(figsize=(17,9))

like_ratio = sns.barplot(data=ratio_with_all,

                         x= 'category',

                         y= 'divided_by_allcount',

                         order=ratio_with_all.sort_values('divided_by_allcount',ascending=False).index )



like_ratio.set_xticklabels(like_ratio.get_xticklabels(),rotation=45)

like_ratio.set_xlabel('category', fontsize = 20)

like_ratio.set_ylabel('ratio(%)', fontsize = 20)



like_ratio.set_title('divide by all amount in category (liked)', fontsize = 20)

like_ratio.tick_params(labelsize = 18)
hated_video = youtube_raw_data.sort_values('dislikes',ascending=False)[:1000]

plt.figure(figsize=(17,9))

most_hated_video = sns.countplot(data= hated_video,

                                 x= 'category',

                                 order= hated_video.category.value_counts().index)

                                





most_hated_video.set_xticklabels(most_hated_video.get_xticklabels(),rotation=45)

most_hated_video.set_xlabel('category', fontsize = 20)

most_hated_video.set_ylabel('count', fontsize = 20)



most_hated_video.set_title('more hated 1000video', fontsize = 20)

most_hated_video.tick_params(labelsize = 18)


# 카운트 플롯이 아닌 숫자연산을 위하여 다른 데이터 프레임을 다시 만들었습니다. 

ratio_with_all_hated = pd.DataFrame(youtube_raw_data.category.value_counts())

#컬럼 네임 변경

ratio_with_all_hated.columns = ['count']

#카테고리 컬럼에 index로 들어가있는 카테고리를 집어넣음

ratio_with_all_hated['category'] = ratio_with_all_hated.index

#싫어요 순위 1000위 안에 들어가 있는 카테고리만으로 ratio_with_all 변경

ratio_with_all_hated = ratio_with_all_hated.loc[list(hated_video.category.value_counts().index)]

#전체 동영상 수로 1000위 안에 들어가있는 카테고리 동영상 수를 나눈 후 *100을 하여 %값 표현

ratio_with_all_hated['divided_by_allcount'] = (hated_video.category.value_counts()/ratio_with_all['count']) * 100
plt.figure(figsize=(17,9))

hated_ratio = sns.barplot(data=ratio_with_all_hated,

                         x= 'category',

                         y= 'divided_by_allcount',

                         order=ratio_with_all_hated.sort_values('divided_by_allcount',ascending=False).index )



hated_ratio.set_xticklabels(hated_ratio.get_xticklabels(),rotation=45)

hated_ratio.set_xlabel('category', fontsize = 20)

hated_ratio.set_ylabel('ratio(%)', fontsize = 20)



hated_ratio.set_title('divide by all amount in category (hated)', fontsize = 20)

hated_ratio.tick_params(labelsize = 18)
youtube_raw_data['all_likes'] = youtube_raw_data.likes + youtube_raw_data.dislikes
voted_video = youtube_raw_data.sort_values('all_likes',ascending=False)[:1000]

plt.figure(figsize=(17,9))

most_voted_video = sns.countplot(data= voted_video,

                                 x= 'category',

                                 order= voted_video.category.value_counts().index)

                                





most_voted_video.set_xticklabels(most_voted_video.get_xticklabels(),rotation=45)

most_voted_video.set_xlabel('category', fontsize = 20)

most_voted_video.set_ylabel('counts', fontsize = 20)



most_voted_video.set_title('after sum of likes and dislikes', fontsize = 20)

most_voted_video.tick_params(labelsize = 18)
comment_count =youtube_raw_data[youtube_raw_data.comment_count >=youtube_raw_data.comment_count.mean()]

plt.figure(figsize=(17,9))

comment = sns.countplot(data= comment_count,

                                 x= 'category',

                                 order= comment_count.category.value_counts().index)

                                





comment.set_xticklabels(comment.get_xticklabels(),rotation=45)

comment.set_xlabel('category', fontsize = 20)

comment.set_ylabel('count', fontsize = 20)



comment.set_title('More commented category distribution', fontsize = 20)

comment.tick_params(labelsize = 18)