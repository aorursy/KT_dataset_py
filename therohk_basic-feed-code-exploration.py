!cat ../input/news-week-17aug24.csv | cut -d"," -f1 | awk '{print substr($1,1,8)}' | sort -n | uniq -c
!cat ../input/news-week-18aug24.csv | cut -d"," -f1 | awk '{print substr($1,1,8)}' | sort -n | uniq -c
!cat ../input/news-week-17aug24.csv | cut -d"," -f2 | sort | uniq -c | sort -nr | head -499
!cat ../input/news-week-18aug24.csv | cut -d"," -f2 | sort | uniq -c | sort -nr | head -499
!cat ../input/news-week-17aug24.csv ../input/news-week-18aug24.csv | cut -d"," -f1 | awk '{print substr($1,7,4)}' | sort -n | uniq -c