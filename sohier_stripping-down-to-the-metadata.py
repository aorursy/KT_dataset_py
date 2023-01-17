import json

import pandas as pd
def tidy_student(student_id, student_entry):

    temp_df = pd.DataFrame(student_entry)

    temp_df['student_id'] = student_id

    del temp_df['solution']

    return temp_df
students = []

failed_entries = []



with open('../input/output.json') as f_open:

    for i, line in enumerate(f_open):

        try:

            # have to drop trailing ,\n from each middle entry 

            student_data = json.loads(line.strip().rstrip(','))

            students.append(tidy_student(i, student_data))

        except:

            failed_entries.append(i)
print(len(failed_entries))
students = pd.concat(students, axis=0)

df = students
df.rename(columns={'name': 'waypoint_id'}, inplace=True)

df['completedDate'] = pd.to_datetime(df['completedDate'], unit='ms', infer_datetime_format=True)

df['waypoint_id'] = df['waypoint_id'].str.replace(r'^Waypoint: ', '')

name_map = {name: i for i, name in enumerate(df.waypoint_id.unique())}

df.replace(to_replace={'waypoint_id': name_map}, inplace=True)
df.to_csv("code_camp_metadata.csv", index=False)
json.dump(name_map, open('waypoint_map.json', 'w+'))