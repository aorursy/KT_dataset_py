# import in our custom functions

import join_forum_post_info as jfpi

import pylint
# read in the three tables we want from metakaggle

forums_info_df, forum_posts_df, forum_topics_df = jfpi.read_in_forum_tables()
# check column names; if all's well you should have no output

jfpi.check_column_names_forum_forums(forums_info_df)

jfpi.check_column_names_forum_posts(forum_posts_df)

jfpi.check_column_names_forum_topics(forum_topics_df)



# join tablesforums_info_df

posts_and_topics_df = jfpi.join_posts_and_topics(forum_posts_df, forum_topics_df)

forum_posts = jfpi.join_posts_with_forum_title(posts_and_topics_df, forums_info_df)
# check joined file

forum_posts.head()
!pylint join_forum_post_info