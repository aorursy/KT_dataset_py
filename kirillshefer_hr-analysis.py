import psycopg2
import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams["figure.figsize"] = (10.0, 7.0)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

pg_connection = {
    "host": "dsstudents.skillbox.ru",
    "port": 5432,
    "dbname": "human_resources",
    "user": "readonly",
    "password": "*"
}
conn = psycopg2.connect(**pg_connection)
cursor = conn.cursor()
sql_str = "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
cursor.execute(sql_str)
table_columns = [a[0] for a in cursor.fetchall()]
print(table_columns)
hr_dataset_df = pd.read_sql_query('SELECT * FROM hr_dataset;', conn)
hr_dataset_df
employees_df = pd.read_sql_query(
'SELECT DISTINCT id, "Employee Name" FROM hr_dataset WHERE "Employee Name" IS NOT NULL;', conn)
print('Количество работников датасета:', len(employees_df), '\n')

state_df = pd.read_sql_query(
'SELECT DISTINCT state FROM hr_dataset WHERE state IS NOT NULL;', conn)
print('Место рождения (штат) :', state_df['state'].to_list(), '\n')

citizendesc_df = pd.read_sql_query(
'SELECT DISTINCT citizendesc FROM hr_dataset WHERE citizendesc IS NOT NULL;', conn)
print('Статус гражданства:', citizendesc_df['citizendesc'].to_list(), '\n')

racedesc_df = pd.read_sql_query(
'SELECT DISTINCT racedesc FROM hr_dataset WHERE racedesc IS NOT NULL;', conn)
print('Группы рас работников компании:', racedesc_df['racedesc'].to_list(), '\n')

term_reasons_df = pd.read_sql_query(
'SELECT DISTINCT "Reason For Term" FROM hr_dataset WHERE "Reason For Term" IS NOT NULL;', conn)
print('Причины увольнения:', term_reasons_df['Reason For Term'].to_list(), '\n')

employment_status_df = pd.read_sql_query(
'SELECT DISTINCT "Employment Status" FROM hr_dataset WHERE "Employment Status" IS NOT NULL;', conn)
print('Статусы работников:', employment_status_df['Employment Status'].to_list(), '\n')

departments_df = pd.read_sql_query(
'SELECT DISTINCT department FROM hr_dataset WHERE department IS NOT NULL;', conn)
print('Подразделения компании:', departments_df['department'].to_list(), '\n')

positions_df = pd.read_sql_query(
'SELECT DISTINCT position FROM hr_dataset WHERE position IS NOT NULL;', conn)
print('Должности компании в датасете:', positions_df['position'].to_list(), '\n')

employee_source_df = pd.read_sql_query(
'SELECT DISTINCT "Employee Source" FROM hr_dataset WHERE "Employee Source" IS NOT NULL;', conn)
print('Источник информации о вакансии', employee_source_df['Employee Source'].to_list(), '\n')

performance_score_df = pd.read_sql_query(
'SELECT DISTINCT "Performance Score" FROM hr_dataset WHERE "Performance Score" IS NOT NULL;', conn)
print('Эффективность сотрудника', performance_score_df['Performance Score'].to_list(), '\n')
production_staff_df = pd.read_sql_query('SELECT * FROM production_staff', conn)
production_staff_df
recruiting_costs_df = pd.read_sql_query("SELECT * FROM recruiting_costs", conn)
recruiting_costs_df
salary_grid_df = pd.read_sql_query('''
SELECT *
FROM salary_grid
ORDER BY "Hourly Mid" DESC;''', conn)
salary_grid_df
employees_number_by_deps = pd.read_sql_query('''
SELECT "department",
COUNT(*) AS "employees number"
FROM hr_dataset
GROUP BY "department"
ORDER BY "department"
''', conn)
employees_number_by_deps.set_index('department', inplace=True)
employees_number_by_deps
employees_number_by_deps.plot.bar(rot=0)
male_female_by_deps = pd.read_sql_query('''
SELECT "department", "sex",
COUNT(*) AS "employees number"
FROM hr_dataset
GROUP BY "department", "sex"
ORDER BY "department", "sex"
''', conn)
male_female_by_deps
male_female_by_deps = male_female_by_deps.pivot\
(index='department', columns='sex', values='employees number')
male_female_by_deps.plot.bar(stacked=True, rot=0)
salary_by_deps = pd.read_sql_query('''
SELECT "department",
SUM("Pay Rate") AS "salary_exspenses"
FROM hr_dataset
GROUP BY "department"
ORDER BY "department"
''', conn)
salary_by_deps.set_index('department', inplace=True)
salary_by_deps
salary_by_deps.plot.bar(rot=0)
male_female_salary_by_deps = pd.read_sql_query('''
SELECT "department", "sex",
SUM("Pay Rate") AS "salary_exspenses"
FROM hr_dataset
GROUP BY "department", "sex"
ORDER BY "department", "sex"
''', conn)
male_female_salary_by_deps
male_female_salary_by_deps = male_female_salary_by_deps.pivot\
(index='department', columns='sex', values='salary_exspenses')
male_female_salary_by_deps.plot.bar(stacked=True, rot=0)
avg_male_female_salary_by_deps = pd.read_sql_query('''
SELECT "department", "sex",
AVG("Pay Rate") AS "average pay rate"
FROM hr_dataset
GROUP BY "department", "sex"
ORDER BY "department", "sex"
''', conn)
avg_male_female_salary_by_deps
avg_male_female_salary_by_deps = avg_male_female_salary_by_deps.pivot\
(index='department', columns='sex', values='average pay rate')
avg_male_female_salary_by_deps.plot.bar(stacked=True, rot=0)
race_structure_by_deps = pd.read_sql_query('''
SELECT "department", "racedesc",
COUNT(*) AS "employees number"
FROM hr_dataset
GROUP BY "department", "racedesc"
ORDER BY "department", "employees number"
''', conn)
race_structure_by_deps
race_structure_by_deps = race_structure_by_deps.pivot\
(index='department', columns='racedesc', values='employees number')
race_structure_by_deps.plot.bar(stacked=True, rot=0)
experience_efficiency = pd.read_sql_query('''
CREATE TEMPORARY TABLE IF NOT EXISTS work_experience_1 AS
    SELECT position,
    MIN("Days Employed") AS "Experience min",
    AVG("Days Employed") AS "Experience mid",
    MAX("Days Employed") AS "Experience max",
    (MIN("Days Employed") + (MAX("Days Employed") - MIN("Days Employed")) / 3) AS "min_mid line",
    (MAX("Days Employed") - (MAX("Days Employed") - MIN("Days Employed")) / 3) AS "mid_max line",
    COUNT(*) AS "Employees number"
    FROM hr_dataset
    GROUP BY position;
    
CREATE TEMPORARY TABLE IF NOT EXISTS work_experience_2 AS
    SELECT *
    FROM work_experience_1
    WHERE "Employees number" > 1;

SELECT work_experience_2.position,
"Employee Name",
(CASE
    WHEN "Performance Score" = 'Exceptional' THEN 6
    WHEN "Performance Score" = 'Exceeds' THEN 5
    WHEN "Performance Score" = 'Fully Meets' THEN 4
    WHEN "Performance Score" = '90-day meets' THEN 3
    WHEN "Performance Score" = 'Needs Improvement' THEN 2
    WHEN "Performance Score" = 'PIP' THEN 1
    WHEN "Performance Score" = 'N/A- too early to review' THEN 0
END) AS "Performance rate",
(CASE
    WHEN hr_dataset."Days Employed" < work_experience_2."min_mid line" THEN 'min experience'
    WHEN hr_dataset."Days Employed" BETWEEN work_experience_2."min_mid line" AND work_experience_2."mid_max line"\
        THEN 'mid experience'
    WHEN hr_dataset."Days Employed" > work_experience_2."mid_max line" THEN 'max experience'
END) AS "Experience status"
INTO experience_efficiency
FROM hr_dataset
JOIN work_experience_2 ON work_experience_2.position = hr_dataset.position;

SELECT "Experience status",
AVG("Performance rate") AS "Average performance rate"
FROM experience_efficiency
GROUP BY "Experience status"
ORDER BY "Experience status";
''', conn)
experience_efficiency
salary_efficiency = pd.read_sql_query('''
CREATE TEMPORARY TABLE IF NOT EXISTS salary_grid_1 AS
    SELECT position,
    MIN("Pay Rate") AS "Hourly min",
    AVG("Pay Rate") AS "Hourly mid",
    MAX("Pay Rate") AS "Hourly max",
    (MIN("Pay Rate") + (MAX("Pay Rate") - MIN("Pay Rate")) / 3) AS "min_mid line",
    (MAX("Pay Rate") - (MAX("Pay Rate") - MIN("Pay Rate")) / 3) AS "mid_max line",
    COUNT(*) AS "Employees number"
    FROM hr_dataset
    GROUP BY position;

CREATE TEMPORARY TABLE IF NOT EXISTS salary_grid_3 AS
    SELECT *
    FROM salary_grid_1
    WHERE "Employees number" > 1;
    
SELECT salary_grid_3.position,
"Employee Name",
(CASE
    WHEN "Performance Score" = 'Exceptional' THEN 6
    WHEN "Performance Score" = 'Exceeds' THEN 5
    WHEN "Performance Score" = 'Fully Meets' THEN 4
    WHEN "Performance Score" = '90-day meets' THEN 3
    WHEN "Performance Score" = 'Needs Improvement' THEN 2
    WHEN "Performance Score" = 'PIP' THEN 1
    WHEN "Performance Score" = 'N/A- too early to review' THEN 0
END) AS "Performance rate",
(CASE
    WHEN hr_dataset."Pay Rate" < salary_grid_3."min_mid line" THEN 'min salary'
    WHEN hr_dataset."Pay Rate" BETWEEN salary_grid_3."min_mid line" AND salary_grid_3."mid_max line"\
        THEN 'mid salary'
    WHEN hr_dataset."Pay Rate" > salary_grid_3."mid_max line" THEN 'max salary'
END) AS "Salary status"
INTO salary_efficiency
FROM hr_dataset
JOIN salary_grid_3 ON salary_grid_3.position = hr_dataset.position;

SELECT "Salary status",
AVG("Performance rate") AS "Average performance rate"
FROM salary_efficiency
GROUP BY "Salary status"
ORDER BY "Salary status";
''', conn)
salary_efficiency
salary_experience_efficiency = pd.read_sql_query('''
SELECT "Experience status",
"Salary status",
AVG(experience_efficiency."Performance rate") AS "Average performance rate"
FROM experience_efficiency
JOIN salary_efficiency ON salary_efficiency."Employee Name" = experience_efficiency."Employee Name"
GROUP BY "Experience status", "Salary status"
ORDER BY "Experience status", "Salary status";
''', conn)
salary_experience_efficiency.set_index(['Experience status', 'Salary status'])
salary_experience_efficiency = salary_experience_efficiency.pivot\
(index='Experience status', columns='Salary status', values='Average performance rate')

salary_experience_efficiency.plot.bar(stacked=True)
age_efficiency = pd.read_sql_query('''
SELECT
(CASE
    WHEN age BETWEEN 25 AND 30 THEN '25-29'
    WHEN age BETWEEN 30 AND 35 THEN '30-34'
    WHEN age BETWEEN 35 AND 40 THEN '35-39'
    WHEN age BETWEEN 40 AND 45 THEN '40-44'
    WHEN age BETWEEN 45 AND 50 THEN '45-49'
    WHEN age BETWEEN 50 AND 55 THEN '50-54'
    WHEN age BETWEEN 55 AND 60 THEN '55-59'
    WHEN age BETWEEN 60 AND 65 THEN '60-64'
    WHEN age >= 65 THEN '65+'
END) AS "Age group",
AVG(CASE
    WHEN "Performance Score" = 'Exceptional' THEN 6
    WHEN "Performance Score" = 'Exceeds' THEN 5
    WHEN "Performance Score" = 'Fully Meets' THEN 4
    WHEN "Performance Score" = '90-day meets' THEN 3
    WHEN "Performance Score" = 'Needs Improvement' THEN 2
    WHEN "Performance Score" = 'PIP' THEN 1
    WHEN "Performance Score" = 'N/A- too early to review' THEN 0
END) AS "Performance rate",
AVG("Days Employed") AS "Average days employed",
COUNT(*) AS "Group size"
FROM hr_dataset
WHERE "Days Employed" > 365
GROUP BY "Age group"
ORDER BY "Age group";
''', conn)
age_efficiency
sns.lineplot(x='Age group', y='Performance rate', data=age_efficiency)
sns.lineplot(x='Age group', y='Average days employed', data=age_efficiency)
marital_status_efficiency = pd.read_sql_query('''
SELECT sex, maritaldesc,
AVG(CASE
    WHEN "Performance Score" = 'Exceptional' THEN 6
    WHEN "Performance Score" = 'Exceeds' THEN 5
    WHEN "Performance Score" = 'Fully Meets' THEN 4
    WHEN "Performance Score" = '90-day meets' THEN 3
    WHEN "Performance Score" = 'Needs Improvement' THEN 2
    WHEN "Performance Score" = 'PIP' THEN 1
    WHEN "Performance Score" = 'N/A- too early to review' THEN 0
END) AS "Performance rate",
AVG("Days Employed") AS "Average days employed",
COUNT(*) AS "Group size"
FROM hr_dataset
WHERE "Days Employed" > 365
GROUP BY sex, maritaldesc
ORDER BY sex, maritaldesc DESC;
''', conn)
marital_status_efficiency.set_index(['sex', 'maritaldesc'])
marital_status_efficiency_perf = marital_status_efficiency.pivot\
(index='sex', columns='maritaldesc', values='Performance rate')

marital_status_efficiency_perf.plot.bar(stacked=True)
marital_status_efficiency_days = marital_status_efficiency.pivot\
(index='sex', columns='maritaldesc', values='Average days employed')

marital_status_efficiency_days.plot.bar(stacked=True)
race_efficiency = pd.read_sql_query('''
SELECT
racedesc,
AVG(CASE
    WHEN "Performance Score" = 'Exceptional' THEN 6
    WHEN "Performance Score" = 'Exceeds' THEN 5
    WHEN "Performance Score" = 'Fully Meets' THEN 4
    WHEN "Performance Score" = '90-day meets' THEN 3
    WHEN "Performance Score" = 'Needs Improvement' THEN 2
    WHEN "Performance Score" = 'PIP' THEN 1
    WHEN "Performance Score" = 'N/A- too early to review' THEN 0
END) AS "Performance rate",
AVG("Days Employed") AS "Average days employed",
COUNT(*) AS "Group size"
FROM hr_dataset
WHERE "Days Employed" > 365
GROUP BY racedesc
ORDER BY "Performance rate" DESC;
''', conn)
race_efficiency
department_perfomance_1 = pd.read_sql_query('''
CREATE TEMPORARY TABLE IF NOT EXISTS department_hires_table AS
    SELECT "department",
    COUNT(*) AS "Employees hires"
    FROM hr_dataset 
    GROUP BY hr_dataset."department"
    HAVING COUNT(*) >= 5
    ORDER BY "department";

CREATE TEMPORARY TABLE IF NOT EXISTS department_terms_table AS
    SELECT "department",
    COUNT(*) AS "Employees terms"
    FROM hr_dataset
    WHERE "Reason For Term" NOT IN ('N/A - Has not started yet', 'N/A - still employed')
    GROUP BY hr_dataset."department"
    ORDER BY "department";

CREATE TEMPORARY TABLE IF NOT EXISTS department_terms_rate_table AS
    SELECT department_hires_table."department", "Employees hires", "Employees terms",
    (CAST("Employees terms" AS decimal(30, 20)) / "Employees hires" * 100) AS "Terms rate"
    FROM department_hires_table
    JOIN department_terms_table ON department_terms_table."department" = department_hires_table."department"
    ORDER BY "Terms rate";

SELECT "department",
100 - 100 / ((SELECT MAX("Terms rate") FROM department_terms_rate_table) - (SELECT MIN("Terms rate") FROM department_terms_rate_table)) *
("Terms rate" - (SELECT MIN("Terms rate") FROM department_terms_rate_table)) AS "Keep_score"
INTO department_keep_score_table
FROM department_terms_rate_table;

SELECT *
FROM department_keep_score_table
''', conn)
department_perfomance_1
department_perfomance_2 = pd.read_sql_query('''
CREATE TEMPORARY TABLE IF NOT EXISTS department_performance_score_table_1 AS
    SELECT "department",
    SUM(CASE
        WHEN "Performance Score" = 'Exceptional' THEN 6
        WHEN "Performance Score" = 'Exceeds' THEN 5
        WHEN "Performance Score" = 'Fully Meets' THEN 4
        WHEN "Performance Score" = '90-day meets' THEN 3
        WHEN "Performance Score" = 'Needs Improvement' THEN 2
        WHEN "Performance Score" = 'PIP' THEN 1
        WHEN "Performance Score" = 'N/A- too early to review' THEN 0
        END) AS "Sum employee rate",
    COUNT("Employee Name") AS "Number of employees"
    FROM hr_dataset
    GROUP BY "department";

CREATE TEMPORARY TABLE IF NOT EXISTS department_performance_score_table_2 AS
    SELECT "department",
    "Sum employee rate" / "Number of employees" AS "Average department rate"
    FROM department_performance_score_table_1
    WHERE "Number of employees" >= 5;

SELECT "department",
(CASE
    WHEN "Average department rate" = 4 THEN 100
    WHEN "Average department rate" = 3 THEN 50
    WHEN "Average department rate" = 2 THEN 0
END) AS "Performance_score"
INTO department_performance_score_table
FROM department_performance_score_table_2;

SELECT *
FROM department_performance_score_table
''', conn)
department_perfomance_2
department_perfomance = pd.read_sql_query('''
SELECT department_keep_score_table."department",
"Keep_score",
"Performance_score",
"Keep_score" + "Performance_score" AS "Total_score"
FROM department_keep_score_table
JOIN department_performance_score_table
    ON department_performance_score_table."department" = department_keep_score_table."department"
ORDER BY "Total_score" DESC
''', conn)
department_perfomance
turnover_by_deps = pd.read_sql_query('''
CREATE TEMPORARY TABLE IF NOT EXISTS turnover_table_1 AS
    SELECT "department",
    COUNT(*) AS "number of employees"
    FROM hr_dataset
    GROUP BY "department"
    ORDER BY "department";

CREATE TEMPORARY TABLE IF NOT EXISTS turnover_table_2 AS
    SELECT "department",
    COUNT(*) AS "turnover_employees"
    FROM hr_dataset
    WHERE "Reason For Term" NOT IN ('N/A - Has not started yet', 'medical issues', 'return to school',\
        'relocation out of area', 'gross misconduct', 'military', 'retiring', 'N/A - still employed')
    GROUP BY "department"
    ORDER BY "department";

SELECT turnover_table_1."department",
CAST("turnover_employees" AS decimal(30, 20)) / "number of employees" * 100 AS "terms-hires score"
FROM turnover_table_1
JOIN turnover_table_2 ON turnover_table_2."department" = turnover_table_1."department"
ORDER BY "terms-hires score" DESC
''', conn)
turnover_by_deps.set_index('department', inplace=True)
turnover_by_deps
turnover_by_deps.plot.bar(rot=0)
'''
Создадим отдельную таблицу hire_cost_score_table и выгрузим в неё оценку по каждой площадке
Затем уже сделаем запрос на отображение результата по данному пункту.
'''

employment_source_perfomance_1 = pd.read_sql_query('''
SELECT hr_dataset."Employee Source",
(100 - CAST(100 AS decimal(30, 20)) / 1323 * recruiting_costs."Total" / COUNT(*)) AS "Hire_cost_score"
INTO hire_cost_score_table
FROM hr_dataset
FULL JOIN recruiting_costs ON recruiting_costs."Employment Source" = hr_dataset."Employee Source"
GROUP BY hr_dataset."Employee Source", recruiting_costs."Total";

SELECT hr_dataset."Employee Source",
COUNT(*) AS "Employees count",
recruiting_costs."Total",
CAST("Total" AS decimal(30, 20)) / COUNT(*) AS "Hire cost",
100 - CAST(100 AS decimal(30, 20)) / 1323 * "Total" / COUNT(*) AS "Hire_score"
FROM hr_dataset
JOIN recruiting_costs ON recruiting_costs."Employment Source" = hr_dataset."Employee Source"
GROUP BY hr_dataset."Employee Source", recruiting_costs."Total"
ORDER BY "Hire cost", "Employees count" DESC;
       ''', conn)
employment_source_perfomance_1
department_perfomance = pd.read_sql_query('''
SELECT department_keep_score_table."department",
"Keep_score",
"Performance_score",
"Keep_score" + "Performance_score" AS "Total_score"
FROM department_keep_score_table
JOIN department_performance_score_table
    ON department_performance_score_table."department" = department_keep_score_table."department"
ORDER BY "Total_score" DESC
''', conn)
department_perfomance
'''Сначала создадим временные таблицы для промежуточных результатов.
Затем создадим отдельную таблицу keep_score_table и выгрузим в неё оценку по каждой площадке.
После же сделаем запрос на отображение результата по данному пункту'''

employment_source_perfomance_2 = pd.read_sql_query('''
CREATE TEMPORARY TABLE IF NOT EXISTS hires_table AS
    SELECT hr_dataset."Employee Source",
    COUNT(*) AS "Employees hires"
    FROM hr_dataset
    GROUP BY hr_dataset."Employee Source"
    ORDER BY "Employee Source";

CREATE TEMPORARY TABLE IF NOT EXISTS terms_table AS
    SELECT "Employee Source",
    COUNT(*) AS "Employees terms"
    FROM hr_dataset
    WHERE "Reason For Term" NOT IN ('N/A - Has not started yet', 'N/A - still employed')
    GROUP BY hr_dataset."Employee Source"
    ORDER BY "Employee Source";

CREATE TEMPORARY TABLE IF NOT EXISTS terms_rate_table AS
    SELECT hires_table."Employee Source", "Employees hires", "Employees terms",
    (CAST("Employees terms" AS decimal(30, 20)) / "Employees hires" * 100) AS "Terms rate"
    FROM hires_table
    JOIN terms_table ON terms_table."Employee Source" = hires_table."Employee Source"
    ORDER BY "Terms rate";

SELECT "Employee Source",
100 - 100 / ((SELECT MAX("Terms rate") FROM terms_rate_table) - (SELECT MIN("Terms rate") FROM terms_rate_table)) *
("Terms rate" - (SELECT MIN("Terms rate") FROM terms_rate_table)) AS "Keep_score"
INTO keep_score_table
FROM terms_rate_table;

SELECT *,
100 - 100 / ((SELECT MAX("Terms rate") FROM terms_rate_table) - (SELECT MIN("Terms rate") FROM terms_rate_table)) *
("Terms rate" - (SELECT MIN("Terms rate") FROM terms_rate_table)) AS "Keep_score"
FROM terms_rate_table;
       ''', conn)
employment_source_perfomance_2
'''Сначала создадим временную таблицу employee_rate_table, где заменим "буквенную" информацию численной.
Затем создадим отдельную таблицу keep_score_table и выгрузим в неё оценку по каждой площадке.
После же сделаем запрос на отображение результата по данному пункту'''

employment_source_perfomance_3 = pd.read_sql_query('''
CREATE TEMPORARY TABLE IF NOT EXISTS employee_rate_table AS
    SELECT "Employee Source",
    SUM(CASE
        WHEN "Performance Score" = 'Exceptional' THEN 6
        WHEN "Performance Score" = 'Exceeds' THEN 5
        WHEN "Performance Score" = 'Fully Meets' THEN 4
        WHEN "Performance Score" = '90-day meets' THEN 3
        WHEN "Performance Score" = 'Needs Improvement' THEN 2
        WHEN "Performance Score" = 'PIP' THEN 1
        WHEN "Performance Score" = 'N/A- too early to review' THEN 0
    END) AS "Sum employee rate",
    COUNT("Employee Name") AS "Number of employees"
    FROM hr_dataset
    GROUP BY "Employee Source";

CREATE TEMPORARY TABLE IF NOT EXISTS perfomance_rate AS
    SELECT "Employee Source", "Number of employees", "Sum employee rate" / "Number of employees" AS "Average employee rate"
    FROM employee_rate_table
    WHERE "Number of employees" >= 5
    ORDER BY "Average employee rate" DESC, "Number of employees" DESC;

SELECT "Employee Source",
(CASE
    WHEN "Average employee rate" = 4 THEN 100
    WHEN "Average employee rate" = 3 THEN 50
    WHEN "Average employee rate" = 2 THEN 0
END) AS "Perfomance_score"
INTO perfomance_score_table
FROM perfomance_rate;

SELECT *,
(CASE
    WHEN "Average employee rate" = 4 THEN 100
    WHEN "Average employee rate" = 3 THEN 50
    WHEN "Average employee rate" = 2 THEN 0
END) AS "Perfomance_score"
FROM perfomance_rate;
       ''', conn)
employment_source_perfomance_3
'''Сначала создадим временную таблицу groupby_table, где посчитаем кол-во сотрудников,пришедших через площадку.
Затем создадим отдельную таблицу hires_count_score_table и выгрузим в неё оценку по каждой площадке.
После же сделаем запрос на выгрузку результата по данному пункту'''

employment_source_perfomance_4 = pd.read_sql_query('''
CREATE TEMPORARY TABLE IF NOT EXISTS groupby_table AS
    SELECT "Employee Source",
    COUNT(*) AS "Employees count"
    FROM hr_dataset
    GROUP BY "Employee Source"
    ORDER BY "Employees count" DESC;

SELECT 
"Employee Source",
(
CAST(100 AS decimal(30, 20)) /
((SELECT MAX("Employees count") FROM groupby_table) - (SELECT MIN("Employees count") FROM groupby_table)) *
("Employees count" - (SELECT MIN("Employees count") FROM groupby_table))
) AS "Hires_count_score"
INTO hires_count_score_table
FROM groupby_table;

SELECT 
"Employee Source",
"Employees count",
(
CAST(100 AS decimal(30, 20)) /
((SELECT MAX("Employees count") FROM groupby_table) - (SELECT MIN("Employees count") FROM groupby_table)) *
("Employees count" - (SELECT MIN("Employees count") FROM groupby_table))
) AS "Hires_count_score"
FROM groupby_table;
       ''', conn)
employment_source_perfomance_4
'''Объединим все временные таблицы в общую и посчитаем сумму баллов по всем метрикам.
После же сделаем запрос на выгрузку результата по данному пункту'''

employment_source_perfomance_total = pd.read_sql_query('''
SELECT hire_cost_score_table."Employee Source",
"Hire_cost_score", "Keep_score",
"Perfomance_score",
"Hires_count_score",
"Hire_cost_score" + "Keep_score" + "Perfomance_score" + "Hires_count_score" AS "Total_score"
FROM hire_cost_score_table
JOIN keep_score_table ON keep_score_table."Employee Source" = hire_cost_score_table."Employee Source"
JOIN perfomance_score_table ON perfomance_score_table."Employee Source" = hire_cost_score_table."Employee Source"
JOIN hires_count_score_table ON hires_count_score_table."Employee Source" = hire_cost_score_table."Employee Source"
ORDER BY "Total_score" DESC;
       ''', conn)
employment_source_perfomance_total
manager_perfomance_1 = pd.read_sql_query('''
CREATE TEMPORARY TABLE IF NOT EXISTS manager_hires_table AS
    SELECT "Manager Name",
    COUNT(*) AS "Employees hires"
    FROM hr_dataset
    GROUP BY "Manager Name"
    ORDER BY "Manager Name";

CREATE TEMPORARY TABLE IF NOT EXISTS manager_terms_table AS
    SELECT "Manager Name",
    COUNT(*) AS "Employees terms"
    FROM hr_dataset
    WHERE "Reason For Term" NOT IN ('N/A - still employed', 'N/A - Has not started yet')
    GROUP BY "Manager Name"
    ORDER BY "Manager Name";

CREATE TEMPORARY TABLE manager_terms_rate_table AS
    SELECT manager_hires_table."Manager Name",
    CAST("Employees terms" AS decimal(30, 20)) / "Employees hires" * CAST(100 AS decimal(30, 20)) AS "Terms rate"
    FROM manager_hires_table
    INNER JOIN manager_terms_table
        ON manager_terms_table."Manager Name" = manager_hires_table."Manager Name"
    ORDER BY "Terms rate";

SELECT "Manager Name",
(
100 - 100  /
((SELECT MAX("Terms rate") FROM manager_terms_rate_table) - (SELECT MIN("Terms rate") FROM manager_terms_rate_table)) *
("Terms rate" - (SELECT MIN("Terms rate") FROM manager_terms_rate_table))
) AS "Keep_score"
INTO manager_keep_score_table
FROM manager_terms_rate_table
ORDER BY "Keep_score" DESC;


SELECT *
FROM manager_keep_score_table
''', conn)
manager_perfomance_1
manager_perfomance_2 = pd.read_sql_query('''
CREATE TEMPORARY TABLE IF NOT EXISTS manager_performance_score_table_1 AS
    SELECT "Manager Name",
    SUM(CASE
            WHEN "Performance Score" = 'Exceptional' THEN 6
            WHEN "Performance Score" = 'Exceeds' THEN 5
            WHEN "Performance Score" = 'Fully Meets' THEN 4
            WHEN "Performance Score" = '90-day meets' THEN 3
            WHEN "Performance Score" = 'Needs Improvement' THEN 2
            WHEN "Performance Score" = 'PIP' THEN 1
            WHEN "Performance Score" = 'N/A- too early to review' THEN 0
        END) AS "Sum employee rate",
    COUNT(*) AS "Number of employees"
    FROM hr_dataset
    GROUP BY "Manager Name";

CREATE TEMPORARY TABLE IF NOT EXISTS manager_performance_score_table_2 AS
    SELECT "Manager Name",
    "Sum employee rate" / "Number of employees"  AS "Average manager rate"
    FROM manager_performance_score_table_1;

SELECT "Manager Name",
(CASE
    WHEN "Average manager rate" = 4 THEN 100
    WHEN "Average manager rate" = 3 THEN 50
    WHEN "Average manager rate" = 2 THEN 0
END) AS "Performance_score"
INTO manager_performance_score_table
FROM manager_performance_score_table_2
ORDER BY "Performance_score" DESC;

SELECT *
FROM manager_performance_score_table
''', conn)
manager_perfomance_2
manager_perfomance_3 = pd.read_sql_query('''
CREATE TEMPORARY TABLE IF NOT EXISTS managers_employees_score_table_1 AS
    SELECT "Manager Name",
    COUNT(*) AS "Number of employees"
    FROM hr_dataset
    GROUP BY "Manager Name";

SELECT "Manager Name",
(
100 /
((SELECT MAX("Number of employees") FROM managers_employees_score_table_1) -
(SELECT MIN("Number of employees") FROM managers_employees_score_table_1)) *
("Number of employees" - (SELECT MIN("Number of employees") FROM managers_employees_score_table_1))
) AS "Employees_score"
INTO managers_employees_score_table
FROM managers_employees_score_table_1
ORDER BY "Employees_score" DESC;

SELECT *
FROM managers_employees_score_table
''', conn)
manager_perfomance_3
manager_perfomance = pd.read_sql_query('''
SELECT manager_keep_score_table."Manager Name",
"Keep_score", "Performance_score", "Employees_score",
"Keep_score" + "Performance_score" + "Employees_score" AS "Total_score"
FROM manager_keep_score_table
JOIN manager_performance_score_table
    ON manager_performance_score_table."Manager Name" = manager_keep_score_table."Manager Name"
JOIN managers_employees_score_table
    ON managers_employees_score_table."Manager Name" = manager_keep_score_table."Manager Name"
ORDER BY "Total_score" DESC
''', conn)
manager_perfomance
salary_efficiency = pd.read_sql_query('''
CREATE TEMPORARY TABLE IF NOT EXISTS salary_grid_1 AS
    SELECT position,
    MIN("Pay Rate") AS "Hourly min",
    AVG("Pay Rate") AS "Hourly mid",
    MAX("Pay Rate") AS "Hourly max",
    (MIN("Pay Rate") + (MAX("Pay Rate") - MIN("Pay Rate")) / 3) AS "min_mid line",
    (MAX("Pay Rate") - (MAX("Pay Rate") - MIN("Pay Rate")) / 3) AS "mid_max line",
    COUNT(*) AS "Employees number"
    FROM hr_dataset
    GROUP BY position;

CREATE TEMPORARY TABLE IF NOT EXISTS salary_grid_2 AS
    SELECT *
    FROM salary_grid_1
    WHERE "Employees number" > 1;
    
SELECT salary_grid_2.position,
"Employee Name",
(CASE
    WHEN "Performance Score" = 'Exceptional' THEN 6
    WHEN "Performance Score" = 'Exceeds' THEN 5
    WHEN "Performance Score" = 'Fully Meets' THEN 4
    WHEN "Performance Score" = '90-day meets' THEN 3
    WHEN "Performance Score" = 'Needs Improvement' THEN 2
    WHEN "Performance Score" = 'PIP' THEN 1
    WHEN "Performance Score" = 'N/A- too early to review' THEN 0
END) AS "Performance rate",
(CASE
    WHEN hr_dataset."Pay Rate" < salary_grid_2."min_mid line" THEN 'min salary'
    WHEN hr_dataset."Pay Rate" BETWEEN salary_grid_2."min_mid line" AND salary_grid_2."mid_max line"\
        THEN 'mid salary'
    WHEN hr_dataset."Pay Rate" > salary_grid_2."mid_max line" THEN 'max salary'
END) AS "Salary status"
INTO salary_efficiency
FROM hr_dataset
JOIN salary_grid_2 ON salary_grid_2.position = hr_dataset.position;

SELECT "Salary status",
AVG("Performance rate") AS "Average performance rate"
FROM salary_efficiency
GROUP BY "Salary status"
ORDER BY "Salary status";
''', conn)
salary_efficiency
experience_efficiency = pd.read_sql_query('''
CREATE TEMPORARY TABLE IF NOT EXISTS work_experience_1 AS
    SELECT position,
    MIN("Days Employed") AS "Experience min",
    AVG("Days Employed") AS "Experience mid",
    MAX("Days Employed") AS "Experience max",
    (MIN("Days Employed") + (MAX("Days Employed") - MIN("Days Employed")) / 3) AS "min_mid line",
    (MAX("Days Employed") - (MAX("Days Employed") - MIN("Days Employed")) / 3) AS "mid_max line",
    COUNT(*) AS "Employees number"
    FROM hr_dataset
    GROUP BY position;
    
CREATE TEMPORARY TABLE IF NOT EXISTS work_experience_2 AS
    SELECT *
    FROM work_experience_1
    WHERE "Employees number" > 1;

SELECT work_experience_2.position,
"Employee Name",
(CASE
    WHEN "Performance Score" = 'Exceptional' THEN 6
    WHEN "Performance Score" = 'Exceeds' THEN 5
    WHEN "Performance Score" = 'Fully Meets' THEN 4
    WHEN "Performance Score" = '90-day meets' THEN 3
    WHEN "Performance Score" = 'Needs Improvement' THEN 2
    WHEN "Performance Score" = 'PIP' THEN 1
    WHEN "Performance Score" = 'N/A- too early to review' THEN 0
END) AS "Performance rate",
(CASE
    WHEN hr_dataset."Days Employed" < work_experience_2."min_mid line" THEN 'min experience'
    WHEN hr_dataset."Days Employed" BETWEEN work_experience_2."min_mid line" AND work_experience_2."mid_max line"\
        THEN 'mid experience'
    WHEN hr_dataset."Days Employed" > work_experience_2."mid_max line" THEN 'max experience'
END) AS "Experience status"
INTO experience_efficiency
FROM hr_dataset
JOIN work_experience_2 ON work_experience_2.position = hr_dataset.position;

SELECT "Experience status",
AVG("Performance rate") AS "Average performance rate"
FROM experience_efficiency
GROUP BY "Experience status"
ORDER BY "Experience status";
''', conn)
experience_efficiency
term_prediction_df = pd.read_sql_query('''
CREATE TEMPORARY TABLE IF NOT EXISTS term_prediction_table AS
    SELECT hr_dataset.*
    FROM hr_dataset
    JOIN experience_efficiency
        ON experience_efficiency."Employee Name" = hr_dataset."Employee Name"
    JOIN salary_efficiency
        ON salary_efficiency."Employee Name" = hr_dataset."Employee Name"
    WHERE "Days Employed" <= 730 AND
        "Reason For Term" NOT IN ('N/A - Has not started yet', 'medical issues', 'return to school',
        'relocation out of area', 'gross misconduct', 'military', 'retiring');

ALTER TABLE term_prediction_table ADD COLUMN "Target" INT;
UPDATE term_prediction_table SET "Target" = 0;

UPDATE term_prediction_table SET "Target" = 1
WHERE "Days Employed" < 365 AND
      "Reason For Term" <> 'N/A - still employed';

SELECT "sex", "maritaldesc", "racedesc", "department", "position", "Manager Name", "Employee Source", "Performance Score",
(CASE
    WHEN "age" BETWEEN 25 AND 30 THEN '25-29'
    WHEN "age" BETWEEN 30 AND 35 THEN '30-34'
    WHEN "age" BETWEEN 35 AND 40 THEN '35-39'
    WHEN "age" BETWEEN 40 AND 45 THEN '40-44'
    WHEN "age" BETWEEN 45 AND 50 THEN '45-49'
    WHEN "age" BETWEEN 50 AND 55 THEN '50-54'
    WHEN "age" BETWEEN 55 AND 60 THEN '55-59'
    WHEN "age" BETWEEN 60 AND 65 THEN '60-64'
    WHEN "age" >= 65 THEN '65+'
END) AS "Age group", 
(CASE
    WHEN "Days Employed" BETWEEN 0 AND 100 THEN '100-'
    WHEN "Days Employed" BETWEEN 100 AND 200 THEN '100-199'
    WHEN "Days Employed" BETWEEN 200 AND 300 THEN '200-299'
    WHEN "Days Employed" BETWEEN 300 AND 365 THEN '300-365'
    WHEN "Days Employed" >= 365 THEN '365+'
END) AS "Experience", 
"Target"
FROM term_prediction_table
''', conn)
term_prediction_df
from sklearn.preprocessing import LabelEncoder


# создадим функцию, которая будет кодировать данные в столбцах
def transform(df):
    transform_df = df.copy()
    le_dict = {column:LabelEncoder() for column in df.columns}
    for column, le_model in le_dict.items():
        le_dict[column] = le_model.fit(df[column])
        transform_df[column] = le_dict[column].transform(df[column])
    return transform_df, le_dict

# а эта функция будет декодировать данные в столбцах на основе передаваемого датасета и словаря енкодеров
def inverse_transform(transform_df, le_dict):
    inverse_transform_df = transform_df.copy()
    for column, le_model in le_dict.items():
        inverse_transform_df[column] = le_dict[column].inverse_transform(transform_df[column])
    return inverse_transform_df, le_dict

# закодируем данные датасета
df_for_learning = transform(term_prediction_df)[0]
le_dict = transform(term_prediction_df)[1]
import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score


# переведём все данные в числовой формат
# for column in [i for i in term_prediction_df.columns if i != 'Days Employed' and i != 'Pay Rate']:
#     term_prediction_df[column] = pd.factorize(term_prediction_df[column])[0]
# обзначим анализируемые и целевые фичи
x = df_for_learning.drop('Target', 1)
y = df_for_learning['Target']
# выбираем стратифицированную кросс-валидацию для дальнейшего GridSearch
skf = StratifiedKFold(n_splits=5, random_state=0)
# разделим датасет на тренировочную и тестовую выборку
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=0)
# возьмём интерпретируемый класификатор - дерево решений
dtc = DecisionTreeClassifier(random_state=0)
# базовые параметры
dtc_params = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': np.arange(3, 6),}
# создание GridSearch estimator'a
dtc_estimator = GridSearchCV(estimator=dtc, param_grid=dtc_params, cv=skf)
# обучение GS estimator'а на тренировочных данных и нахождение наилучших параметров
dtc_model = dtc_estimator.fit(x_train, y_train)
# предсказание обученной модели
y_pred = dtc_model.predict(x_test)
# процент правильных ответов
accuracy_score(y_test, y_pred)
dtc_estimator.best_params_
dtc = DecisionTreeClassifier(criterion='gini', max_depth=3, splitter='random', random_state=0)
dtc_model = dtc.fit(x_train, y_train)
y_pred = dtc_model.predict(x_test)
# процент правильных ответов
accuracy_score(y_test, y_pred)
from sklearn import tree
import graphviz

def print_graph(data):
    dot_data = tree.export_graphviz(data, out_file=None,
                                    feature_names=x.columns,  
                                    class_names='Target',  
                                    filled=True, rounded=True,
                                    special_characters=True)  
    return graphviz.Source(dot_data)

print_graph(dtc_model)
columns = ['Experience', 'department', 'Manager Name']
for column in columns:
    classes = le_dict[column].classes_
    print(column)
    print({name:code for name, code in enumerate(classes)})
    print()