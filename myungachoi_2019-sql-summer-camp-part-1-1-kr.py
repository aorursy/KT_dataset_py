# Google Cloud 에서 BigQuery 라이브러리를 가져옵니다.

# 동시에 Kaggle을 통한 BigQuery 사용자 인증절차와 Kaggle의 public dataset을 BigQuery에서 사용하기 위한 설정이 진행됩니다.



from google.cloud import bigquery
# Create a "Client" object

# BigQuery의 Cleint object는 BigQuery와 사용자 사이에서 메신저 역할을 합니다.

# 이 예제를 위해 client라는 이름으로 Client object instance를(복제?!) 생성합니다.



client = bigquery.Client()
# Construct a reference to the "hacker_news" dataset

# 메신저 client는 모든 BigQuery public dataset이 있는 곳(=bigquery-public-data)에 가서

# hacker_news라는 dataset을 찾은 후 그 주소를(위치를) dataset_ref에 저장합니다.



dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



# API request - fetch the dataset

# dataset_ref가 가리키는 곳을 찾아가기 위한 지도(찾아가는 방법)을 dataset에 저장합니다.



dataset = client.get_dataset(dataset_ref)
# List all the tables in the "hacker_news" dataset

# 지도를 들고 찾아간 곳에 있는 hacker news dataset이 어떤 내용을 가지고 있는지 알아보기 위해

# 해당되는 위치의 dataset 내용을 들여다보고 그 안에 있는 표들을 리스트 형태로 가져옵니다.



tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset

# 가져온 표들의 실제 내용(이름)을 print를 사용해 출력합니다.



for table in tables:

    print(table.table_id)
# Construct a reference to the "full" table

# 출력된 4개의 표 이름 중에 full이라는 표의 주소를 table_ref에 저장합니다.



table_ref = dataset_ref.table("full")



# API request - fetch the table

# table_ref가 가리키는 곳을 찾아가기 위한 지도(찾아가는 방법)을 table에 저장합니다.



table = client.get_table(table_ref)
# Print information on all the columns in the "full" table in the "hacker_news" dataset

# full 이라는 표의 구성 정보를 출력합니다. 실제 대이타를 출력하는 것은 아닙니다.

table.schema
# Preview the first five lines of the "full" table

# full tanle의 처음 다섯행을 가져와 테이블 서식대로 출력합니다

client.list_rows(table, max_results=5).to_dataframe()
# Preview the first five entries in the "by" column of the "full" table

# 특정 열에서 처음 다섯행을 가져와 데이블 서식대로 출력합니다.

client.list_rows(table, selected_fields=table.schema[:1], max_results=5).to_dataframe()