import sqlite3

import pandas as pd
conn = sqlite3.connect("/kaggle/input/podcastreviews/database.sqlite")
pd.read_sql("select avg(rating) from reviews", conn)
pd.read_sql("select category, avg(rating), count(*) from reviews join categories using (podcast_id) group by 1 order by 2 desc", conn)
! sqlite3 /kaggle/input/podcastreviews/database.sqlite 'select count(*) from reviews'
! wget --quiet https://github.com/soaxelbrooke/phrase/releases/download/0.3.6/phrase-0.3.6-x86_64-unknown-linux-gnu.tar.gz
! tar -xzvf phrase-0.3.6-x86_64-unknown-linux-gnu.tar.gz
! ./phrase count -h
! sqlite3 --header --csv /kaggle/input/podcastreviews/database.sqlite 'select category, rating, title, content from reviews join categories using (podcast_id)' | ./phrase count -m csv - --labelfield category  --labelfield rating --textfield title --textfield content
! LANG=en MIN_COUNT=10 ./phrase export
! ./phrase show -n 10