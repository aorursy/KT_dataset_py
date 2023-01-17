import csv
import pulp # library used for linear programming
# PREPARATION: SETUP FUNCTIONS FOR READING/WRITING CSV
#------------------------------------------------------

def read_in_csv_to_list(filename, headers=False):
    """
    :param filename: A string indicating the file name (or relative path)
    :param headers: Whether the first row of the CSV contains headers (default False)
    :return: A list where each item in the list corresponds to a row in the
             CSV (each item in turn is a list corresponding to columns)
    """
    start_row = 0
    if headers:
        start_row = 1
    with open(filename, 'r') as f:
        data = [row for row in csv.reader(f.read().splitlines())]
        return data[start_row:]

def write_list_to_csv(filename, data, headers=None):
    """
    :param filename: A string indicating the file name (or relative path)
    :param headers: A list of headers to write to the first row of the file
    :param data: A list of lists to write to the CSV file
    :return: A list where each item in the list corresponds to a row in the
             CSV (each item in turn is a list corresponding to columns)
    """
    with open(filename, 'wb') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        if headers:
            csv_writer.writerow(headers)
        for row in data:
            csv_writer.writerow(row)
# STEP 1: LOAD IN THE RAW DATA
student_choice_data = read_in_csv_to_list('student_choices.csv', headers=True)
student_choice_data
[['Boro', 'Mimi', 'I-fight', 'LFR', 'Mastermind', 'Eagle'],
 ['Sajid', 'Hasin', 'E-Board', 'LFR', 'LAN', 'I-fight'],
 ['Azim ', 'Sarwar', 'Mastermind', 'I-fight', 'Braniac', 'LFR'],
 ['Choto', 'Mimi', 'LFR', 'I-fight', 'Suduko', 'Rubiks Cube'],
 ['Humyara', 'Raka', 'LFR', 'I-fight', 'Braniac', 'Robot Race'],
 ['Albab', 'Amin', 'I-fight', 'Suduko', 'Eagle', 'Solidworks'],
 ['Mezbah', 'Uddin', 'I-fight', 'Solidworks', 'Braniac', 'LAN'],
 ['Sufi', 'sifat', 'I-fight', 'Suduko', 'Rubiks Cube', 'LFR'],
 ['Fahmida ', 'Faiza', 'I-fight', 'LFR', 'Eagle', 'Rubiks Cube'],
 ['Ishrat', 'Humayara', 'Rubiks Cube', 'LAN', 'I-fight', 'Eagle'],
 ['Sayonyee', 'Roy', 'I-fight', 'LFR', 'Robot Race', 'Mastermind'],
 ['Mukit ', 'Emon', 'Eagle', 'Mastermind', 'Solidworks', 'E-Board'],
 ['Sk', 'Fatema', 'Rubiks Cube', 'Mastermind', 'LAN', 'I-fight'],
 ['Jakir', 'Joti', 'Mastermind', 'LFR', 'Solidworks', 'E-Board'],
 ['Mehrab', 'Niloy', 'I-fight', 'LFR', 'Eagle', 'Rubiks Cube'],
 ['Saikat', 'Das', 'Braniac', 'I-fight', 'LAN', 'Suduko'],
 ['S.U.M', 'Neelim', 'LAN', 'I-fight', 'Suduko', 'Braniac'],
 ['P. Magnet', 'Sohan', 'Mastermind', 'LFR', 'Suduko', 'Solidworks'],
 ['Chacha', 'Tamim', 'I-fight', 'Solidworks', 'Rubiks Cube', 'LAN'],
 ['Ferdous', 'Shawon', 'I-fight', 'LAN', 'Eagle', 'Suduko'],
 ['Professor', 'Akib', 'I-fight', 'Rubiks Cube', 'LAN', 'Braniac'],
 ['Smart', 'Bappi', 'I-fight', 'Mastermind', 'E-Board', 'LFR'],
 ['Kj', 'Jakir', 'I-fight', 'Robot Race', 'LAN', 'Braniac'],
 ['Jamai', 'Oscar', 'I-fight', 'Suduko', 'Robot Race', 'LAN'],
 ['P. Magnet', 'Anas', 'I-fight', 'LAN', 'Robot Race', 'Suduko']]

# STEP 2: SET CONSTANT VARIABLES
#------------------------------------------------------
ACTIVITIES_LIST = ['LAN', 'I-fight', 'Eagle',
                   'Braniac', 'Rubiks Cube', 'LFR',
                   'Suduko', 'Robot Race', 'E-Board',
                   'Mastermind', 'Solidworks']

NUM_ACTIVITIES = len(ACTIVITIES_LIST)
NUM_PERIODS = 3 # there are three sessions
NUM_STUDENTS = len(student_choice_data) # this can change based on the input data, but remains constant once here
MIN_NUM_IN_ACTIVITY = 2
MAX_NUM_IN_ACTIVITY = 5

# STEP 3: LOAD STUDENT PREFERENCES INTO A USEFUL FORMAT


# assign each activity a number as an activity_id
activities_dict = {activity: n for n, activity in enumerate(ACTIVITIES_LIST)}
activities_dict_reverse = {n: activity for n, activity in enumerate(ACTIVITIES_LIST)}

# assign each student a number as an ID and setup a dictionary with the number as key
# and a dictionary with choices (ordered list of activity_ids by preference) and student name
student_info_dicts = {}
for i, row in enumerate(student_choice_data):
    first_name, last_name, choice1, choice2, choice3, choice4 = row
    name =  last_name + ', ' + first_name
    choices = [choice1, choice2, choice3, choice4]
    student_info_dicts[i] = {'choices': [activities_dict[c] for c in choices],
                             'choices_full_names': choices,
                             'name': name,
                             'assignments': []}

# STEP 4: DEFINE THE PROBLEM TYPE

# define the problem as an optimization to maximize the objective function
prob = pulp.LpProblem("StudentClass", pulp.LpMaximize)

# STEP 5: SETUP DECISION VARIABLES
#------------------------------------------------------

# start off by making a simple list of coordinates for decision variables, in the format (i, j)
# where "i" indicates the  student number and "j" indicates the activity number:
decision_var_matrix = []
for i in range(NUM_STUDENTS):
    for j in range(NUM_ACTIVITIES):
        decision_var_matrix.append((i, j))
# at this point, decision_var_matrix = [(0, 0), (0, 1) ... (0, 11), (1, 0), (1, 1), etc...]

# now setup a set of decision variables, one for each period
decision_vars_list = []
for i in range(NUM_PERIODS):
    variable_type_name = 'period_%s_decision_variable' % (i + 1)
    decision_vars_list.append(pulp.LpVariable.dicts(name=variable_type_name,
                                                    indexs=decision_var_matrix,
                                                    lowBound=0,
                                                    upBound=1,
                                                    cat=pulp.LpInteger))
    # note: For continuous variables, set "cat" (i.e. category) to pulp.LpContinuous instead

# We now have one set of decision variables for each period. Think of each set of decision
# variables as an m-by-n matrix of binary (1 or 0) values where 'm' is the number of students and
# 'n' is the number of activities. For the given period, a value of 1 indicates that the student
# is in that activity in that period and a value of 0 means they are not.

# STEP 6: SETUP CONSTRAINTS
#------------------------------------------------------

# CONSTRAINT 1: each student must be in one and only one activity per period
for i in range(NUM_STUDENTS):
    for k in range(NUM_PERIODS):
        vars_to_sum = [decision_vars_list[k][(i, j)] for j in range(NUM_ACTIVITIES)]
        prob += pulp.lpSum(vars_to_sum) == 1

# CONSTRAINT 2: a student cannot repeat an activity
for i in range(NUM_STUDENTS):
    for j in range(NUM_ACTIVITIES):
        vars_to_sum = [decision_vars_list[k][(i, j)] for k in range(NUM_PERIODS)]
        prob += pulp.lpSum(vars_to_sum) <= 1

# CONSTRAINT 3: each activity must have a minimum number of students
for j in range(NUM_ACTIVITIES):
    for k in range(NUM_PERIODS):
        vars_to_sum = [decision_vars_list[k][(i, j)] for i in range(NUM_STUDENTS)]
        prob += pulp.lpSum(vars_to_sum) >= MIN_NUM_IN_ACTIVITY

# CONSTRAINT 4: each activity can only have up to a maximum number of students
for j in range(NUM_ACTIVITIES):
    for k in range(NUM_PERIODS):
        vars_to_sum = [decision_vars_list[k][(i, j)] for i in range(NUM_STUDENTS)]
        prob += pulp.lpSum(vars_to_sum) <= MAX_NUM_IN_ACTIVITY

# CONSTRAINT 5: each student must get 3 of their 4 choices
for i in range(NUM_STUDENTS):
    # remember, student_info_dicts[i]['choices'] looks like [x1, x2, x3, x4] where
    # x1...x4 are the activity numbers for the student's first through fourth
    # choices respectively
    vars_to_sum = [decision_vars_list[k][(i, j)] for j in student_info_dicts[i]['choices']
                                                 for k in range(NUM_PERIODS)]
    prob += pulp.lpSum(vars_to_sum) == 3

# STEP 6: SETUP OBJECTIVE FUNCTION
#------------------------------------------------------

# The objective function is gonna give 1000 "points" for each 1st choice match made,
# 100 for each second choice, 10 for each third choice and 1 for each fourth choice.
# To do this, for each set of decision variables, we iterate through each student
# and get their choices. For each choice, the corresponding decision variables are
# multiplied by the appropriate number of points (1000, 100, 10 or 1) and added to
# the objective_function_parts list. When the list is done being assembled, we set
# the objective function by running lpSum on the list. Note that with this objective
# function, there is no preference on _when_ a student does a preferred activity.
# For example, the same number of points are rewarded whether the student does their
# first choice activity in the first period or the last period.
objective_function_parts = []
for i in range(NUM_STUDENTS):
    for k in range(NUM_PERIODS):
        choice1, choice2, choice3, choice4 = student_info_dicts[i]['choices']
        objective_function_parts.append([decision_vars_list[k][(i, choice1)]*1000])
        objective_function_parts.append([decision_vars_list[k][(i, choice2)]*100])
        objective_function_parts.append([decision_vars_list[k][(i, choice3)]*10])
        objective_function_parts.append([decision_vars_list[k][(i, choice4)]*1])

prob += pulp.lpSum(objective_function_parts)

# STEP 7: SOLVE THE LP
solution_found = prob.solve()

# check that a solution was found (anything other than solution_found == 1
# indicates some issue with finding a solution that fits constraints)
assert solution_found == 1,"solution not found"
print("Objective value:", pulp.value(prob.objective))


# STEP 8: EXTRACT THE SOLUTION INTO A USEFUL FORMULA
# In this case, let's try to take the solution and put it into a CSV output file. While we're at
# it, we'll collect some data to see how we did

for period in range(NUM_PERIODS):
    for i in range(NUM_STUDENTS):
        for j in range(NUM_ACTIVITIES):
            if decision_vars_list[period][(i,j)].varValue == 1:
                student_info_dicts[i]['assignments'].append(activities_dict_reverse[j])

results = []
for i in range(NUM_STUDENTS):
    student_dict = student_info_dicts[i]
    name = student_dict['name']
    choices = student_dict['choices_full_names']
    choice1, choice2, choice3, choice4 = choices
    assignments = student_dict['assignments']
    assignment1, assignment2, assignment3 = assignments
    row = [name, choice1, choice2, choice3, choice4, assignment1, assignment2, assignment3]
    results.append(row)

headers = ['name', 'choice1', 'choice2', 'choice3', 'choice4',
           'assignment1', 'assignment2' ,'assignment3', 'assignment4']
write_list_to_csv('assignment_outputs.csv', data=results, headers=headers)


# STEP 9: SANITY CHECK OUR RESULTS
#------------------------------------------------------

total_got_choice1 = 0
total_got_choice2 = 0
total_got_choice3 = 0
total_got_choice4 = 0
# due to constraint 5, we know that every student gets 3 of 4 choices, but it doesn't
# hurt to check that over here
total_got_3_of_4 = 0

for i in range(NUM_STUDENTS):
    choices = student_info_dicts[i]['choices_full_names']
    choice1, choice2, choice3, choice4 = choices
    assignments = student_info_dicts[i]['assignments']

    got_choice1, got_choice2, got_choice3, got_choice4 = 0, 0, 0, 0
    if choice1 in assignments: got_choice1 = 1
    if choice2 in assignments: got_choice2 = 1
    if choice3 in assignments: got_choice3 = 1
    if choice4 in assignments: got_choice4 = 1
    if (got_choice1 + got_choice2 + got_choice3 + got_choice4) == 3:
        total_got_3_of_4 += 1
    total_got_choice1 += got_choice1
    total_got_choice2 += got_choice2
    total_got_choice3 += got_choice3
    total_got_choice4 += got_choice4

print("percent who got choice 1:", float(total_got_choice1) / NUM_STUDENTS * 100)
print("percent who got choice 2:", float(total_got_choice2) / NUM_STUDENTS * 100)
print("percent who got choice 3:", float(total_got_choice3) / NUM_STUDENTS * 100)
print("percent who got choice 4:", float(total_got_choice4) / NUM_STUDENTS * 100)
print("percent who got 3 of 4 choices:", float(total_got_3_of_4) / NUM_STUDENTS * 100)