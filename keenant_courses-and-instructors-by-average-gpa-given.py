import pandas as pd # reading files
import sqlite3 # sqlite database
from shutil import copyfile # copy database func

# move the database into a writeable format
copyfile('../input/database.sqlite3', './database.sqlite3')

# open connection
conn = sqlite3.connect('./database.sqlite3')
conn
pd.read_sql("""
  SELECT
    course_offering_uuid,
    section_number,
    (4.0 * a_count + 3.5 * ab_count + 3.0 * b_count + 2.5 * bc_count + 2 * c_count + 1 * d_count) / (a_count + ab_count + b_count + bc_count + c_count + d_count + f_count) AS gpa
  FROM grade_distributions
  LIMIT 5
""", conn)
# we add a new view
c = conn.cursor()
c.execute("DROP VIEW IF EXISTS section_gpas")
c.execute("""
  CREATE VIEW
  section_gpas (course_offering_uuid, section_number, gpa, num_grades)
  AS
  SELECT
    course_offering_uuid,
    section_number,
    (4.0 * a_count + 3.5 * ab_count + 3.0 * b_count + 2.5 * bc_count + 2 * c_count + 1 * d_count) / (a_count + ab_count + b_count + bc_count + c_count + d_count + f_count) AS gpa,
    a_count + ab_count + b_count + bc_count + c_count + d_count + f_count AS num_grades
  FROM grade_distributions
""")
# and now we can query it...
pd.read_sql("SELECT * FROM section_gpas LIMIT 5", conn)
pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
pd.read_sql("""
  SELECT *
  FROM instructors i
  JOIN teachings t ON i.id = t.instructor_id
  JOIN sections s ON s.uuid = t.section_uuid
  JOIN section_gpas gpas ON gpas.course_offering_uuid = s.course_offering_uuid AND gpas.section_number = s.number
  LIMIT 5
""", conn)
pd.read_sql("""
  SELECT 
    i.id, 
    i.name,
    SUM(gpas.gpa * gpas.num_grades) / SUM(gpas.num_grades) as avg_gpa
  FROM instructors i
  JOIN teachings t ON i.id = t.instructor_id
  JOIN sections s ON s.uuid = t.section_uuid
  JOIN section_gpas gpas ON gpas.course_offering_uuid = s.course_offering_uuid AND gpas.section_number = s.number
  GROUP BY i.id
  ORDER BY avg_gpa DESC
  LIMIT 10
""", conn)
pd.read_sql("""
  SELECT 
    i.id, 
    i.name,
    SUM(gpas.gpa * gpas.num_grades) / SUM(gpas.num_grades) as avg_gpa,
    AVG(gpas.num_grades), 
    SUM(gpas.num_grades)
  FROM instructors i
  JOIN teachings t ON i.id = t.instructor_id
  JOIN sections s ON s.uuid = t.section_uuid
  JOIN section_gpas gpas ON gpas.course_offering_uuid = s.course_offering_uuid AND gpas.section_number = s.number
  GROUP BY i.id
  ORDER BY avg_gpa DESC
  LIMIT 10
""", conn)
pd.read_sql("""
  SELECT 
    i.id, 
    i.name, 
    SUM(gpas.gpa * gpas.num_grades) / SUM(gpas.num_grades) as avg_gpa,
    AVG(gpas.num_grades) as avg_num_grades, 
    SUM(gpas.num_grades) as total_num_grades
  FROM instructors i
  JOIN teachings t ON i.id = t.instructor_id
  JOIN sections s ON s.uuid = t.section_uuid
  JOIN section_gpas gpas ON gpas.course_offering_uuid = s.course_offering_uuid AND gpas.section_number = s.number
  GROUP BY i.id
  HAVING avg_num_grades >= 30 AND total_num_grades >= 250
  ORDER BY avg_gpa DESC
  LIMIT 15
""", conn)
pd.read_sql("""
  SELECT 
    i.id,
    i.name,
    SUM(gpas.gpa * gpas.num_grades) / SUM(gpas.num_grades) as avg_gpa,
    AVG(gpas.num_grades) as avg_num_grades, 
    SUM(gpas.num_grades) as total_num_grades
  FROM instructors i
  JOIN teachings t ON i.id = t.instructor_id
  JOIN sections s ON s.uuid = t.section_uuid
  JOIN section_gpas gpas ON gpas.course_offering_uuid = s.course_offering_uuid AND gpas.section_number = s.number
  GROUP BY i.id
  HAVING avg_num_grades >= 30 AND total_num_grades >= 250
  ORDER BY avg_gpa ASC
  LIMIT 15
""", conn)
pd.read_sql("""
  SELECT 
    c.uuid, 
    c.name,
    GROUP_CONCAT(DISTINCT subjects.abbreviation) as subjects,
    c.number,
    SUM(gpas.gpa * gpas.num_grades) / SUM(gpas.num_grades) as avg_gpa,
    AVG(gpas.num_grades) as avg_num_grades, 
    SUM(gpas.num_grades) as total_num_grades
  FROM courses c
  JOIN course_offerings co ON co.course_uuid = c.uuid
  JOIN subject_memberships sm ON sm.course_offering_uuid = co.uuid
  JOIN subjects ON sm.subject_code = subjects.code
  JOIN sections s ON s.course_offering_uuid = co.uuid
  JOIN section_gpas gpas ON gpas.course_offering_uuid = s.course_offering_uuid AND gpas.section_number = s.number
  GROUP BY c.uuid
  HAVING total_num_grades >= 250
  ORDER BY avg_gpa ASC
  LIMIT 15
""", conn)
pd.read_sql("""
  SELECT
    subjects.name,
    subjects.abbreviation,
    SUM(gpas.gpa * gpas.num_grades) / SUM(gpas.num_grades) as avg_gpa,
    AVG(gpas.num_grades) as avg_num_grades, 
    SUM(gpas.num_grades) as total_num_grades
  FROM subjects
  JOIN subject_memberships sm ON sm.subject_code = subjects.code
  JOIN course_offerings co ON co.uuid = sm.course_offering_uuid
  JOIN sections s ON s.course_offering_uuid = co.uuid
  JOIN section_gpas gpas ON gpas.course_offering_uuid = s.course_offering_uuid AND gpas.section_number = s.number
  GROUP BY subjects.code
  HAVING total_num_grades > 0
  ORDER BY avg_gpa ASC
  LIMIT 30
""", conn)