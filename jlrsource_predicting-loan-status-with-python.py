import matplotlib.pyplot as plt

import numpy as np

import sqlite3

from sklearn import tree

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



def sql_query(s):

    """Return results for a SQL query.



    Arguments:

    s (str) -- SQL query string



    Returns:

    (list) -- SQL query results

    """

    conn = sqlite3.connect("../input/database.sqlite")

    c = conn.cursor()

    c.execute(s)

    result = c.fetchall()

    conn.close()

    return result



def print_details():

    """Print database details including table names and the number of rows.

    """

    table_names = sql_query("SELECT name FROM sqlite_master " +

                            "WHERE type='table' " +

                            "ORDER BY name;")[0][0]

    print("Names of tables in SQLite database: {0}".format(table_names))

    num_rows = sql_query("SELECT COUNT(*) FROM loan;")[0][0]

    print("Number of records in table: {0}".format(num_rows))



def print_column_names():

    """Print the column names in the 'loan' table.

    Note that the "index" column name is specific to Python and is not part of

    the original SQLite database.

    """

    conn = sqlite3.connect("../input/database.sqlite")

    conn.row_factory = sqlite3.Row

    c = conn.cursor()

    c.execute("SELECT * FROM loan LIMIT 2;")

    r = c.fetchone()

    i = 1

    print("Column names:")

    for k in r.keys():

        print("{0:d}\t{1}".format(i, k))

        i += 1

    conn.close()



print_details()

print_column_names()
emp_length_dict = {'n/a':0,

                   '< 1 year':0,

                   '1 year':1,

                   '2 years':2,

                   '3 years':3,

                   '4 years':4,

                   '5 years':5,

                   '6 years':6,

                   '7 years':7,

                   '8 years':8,

                   '9 years':9,

                   '10+ years':10}



home_ownership_dict = {'MORTGAGE':0,

                       'OWN':1,

                       'RENT':2,

                       'OTHER':3,

                       'NONE':4,

                       'ANY':5}



features_dict = {'loan_amnt':0,

                 'int_rate':1,

                 'annual_inc':2,

                 'delinq_2yrs':3,

                 'open_acc':4,

                 'dti':5,

                 'emp_length':6,

                 'funded_amnt':7,

                 'tot_cur_bal':8,

                 'home_ownership':9}



def get_data(s):

    """Return features and targets for a specific search term.



    Arguments:

    s (str) -- string to search for in loan "title" field



    Returns:

    (list of lists) -- [list of feature tuples, list of targets]

         (features) -- [(sample1 features), (sample2 features),...]

           (target) -- [sample1 target, sample2 target,...]

    """

    data = sql_query("SELECT " +

                     "loan_amnt,int_rate,annual_inc," +

                     "loan_status,title,delinq_2yrs," +

                     "open_acc,dti,emp_length," +

                     "funded_amnt,tot_cur_bal,home_ownership " +

                     "FROM loan " +

                     "WHERE application_type='INDIVIDUAL';")

    features_list = []

    target_list = []

    n = 0   # counter, number of total samples

    n0 = 0  # counter, number of samples with target=0

    n1 = 0  # counter, number of samples with target=1

    for d in data:

        # d[0] (loan_amnt)   -- must have type 'float'

        # d[1] (int_rate)    -- must have type 'str'

        # d[2] (annual_inc)  -- must have type 'float'

        # d[3] (loan_status) -- must have type 'str'

        # d[4] (title)       -- must have type 'str'

        # d[5] (delinq_2yrs) -- must have type 'float'

        # d[6] (open_acc)    -- must have type 'float'

        # d[7] (dti)         -- must have type 'float'

        # d[8] (emp_length)  -- must have type 'str'

        # d[9] (funded_amnt) -- must have type 'float'

        # d[10] (tot_cur_bal) -- must have type 'float'

        # d[11] (home_ownership) -- must have type 'str'

        test0 = isinstance(d[0], float)

        test1 = isinstance(d[1], str)

        test2 = isinstance(d[2], float)

        test3 = isinstance(d[3], str)

        test4 = isinstance(d[4], str)

        test5 = isinstance(d[5], float)

        test6 = isinstance(d[6], float)

        test7 = isinstance(d[7], float)

        test8 = isinstance(d[8], str)

        test9 = isinstance(d[9], float)

        test10 = isinstance(d[10], float)

        if (test0 and test1 and test2 and test3 and test4 and test5 and

            test6 and test7 and test8 and test9 and test10):

            # Ensure that "int_rate" string value can be converted to float

            try:

                d1_float = float(d[1].replace("%", ""))

            except:

                continue

            # Ensure that "emp_length" string value is in dict

            try:

                e = emp_length_dict[d[8]]

            except:

                print("Error e")

                continue

            # Ensure that "home_ownership" string value is in dict

            try:

                h = home_ownership_dict[d[11]]

            except:

                print("Error h")

                continue

            # Set "title" string to lowercase for search purposes

            if s.lower() in d[4].lower():

                if d[3] == 'Fully Paid' or d[3] == 'Current':

                    target = 0  # Define target value as 0

                    n += 1

                    n0 += 1

                elif 'Late' in d[3] or d[3] == 'Charged Off':

                    target = 1  # Define target value as 1

                    n += 1

                    n1 += 1

                else:

                    continue

                # Define features tuple:

                # (loan_amnt, int_rate, annual_inc)

                features = (d[0],

                            float(d[1].replace("%", "")),

                            d[2],

                            d[5],

                            d[6],

                            d[7],

                            emp_length_dict[d[8]],

                            d[9],

                            d[10],

                            home_ownership_dict[d[11]])

                features_list.append(features)

                target_list.append(target)

        else:

            pass

    print("----------------------------------------")

    print(s)

    print("----------------------------------------")

    print("Total number of samples: {0}".format(n))

    print("% of all samples with target=0: {0:3.4f}%".format(100*n0/(n0+n1)))

    print("% of all samples with target=1: {0:3.4f}%".format(100*n1/(n0+n1)))

    print("")

    result = [features_list, target_list]

    return result



def create_scatter_plot(x0_data, y0_data,

                        x1_data, y1_data,

                        pt, pa,

                        x_label, y_label,

                        axis_type):

    plt.figure(num=2, figsize=(8, 8))

    ax = plt.gca()

    ax.set_axis_bgcolor("#BBBBBB")

    ax.set_axisbelow(True)

    plt.subplots_adjust(bottom=0.1, left=0.15, right=0.95, top=0.95)

    plt.title(pt, fontsize=16)

    plt.axis(pa)

    plt.xlabel(x_label, fontsize=16)

    plt.ylabel(y_label, fontsize=16)

    plt.xticks(fontsize=16)

    plt.yticks(fontsize=16)

    if axis_type == 'semilogx':

        plt.semilogx(x0_data, y0_data, label='0: "Fully Paid" or "Current"',

                     linestyle='None', marker='.', markersize=8,

                     alpha=0.5, color='b')

        plt.semilogx(x1_data, y1_data, label='1: "Late" or "Charged Off"',

                     linestyle='None', marker='.', markersize=8,

                     alpha=0.5, color='r')

    elif axis_type == 'semilogy':

        plt.semilogy(x0_data, y0_data, label='0: "Fully Paid" or "Current"',

                     linestyle='None', marker='.', markersize=8,

                     alpha=0.5, color='b')

        plt.semilogy(x1_data, y1_data, label='1: "Late" or "Charged Off"',

                     linestyle='None', marker='.', markersize=8,

                     alpha=0.5, color='r')

    elif axis_type == "loglog":

        plt.loglog(x0_data, y0_data, label='0: "Fully Paid" or "Current"',

                   linestyle='None', marker='.', markersize=8,

                   alpha=0.5, color='b')

        plt.loglog(x1_data, y1_data, label='1: "Late" or "Charged Off"',

                   linestyle='None', marker='.', markersize=8,

                   alpha=0.5, color='r')

    else:

        plt.plot(x0_data, y0_data, label='0: "Fully Paid" or "Current"',

                 linestyle='None', marker='.', markersize=8,

                 alpha=0.5, color='b')

        plt.plot(x1_data, y1_data, label='1: "Late" or "Charged Off"',

                 linestyle='None', marker='.', markersize=8,

                 alpha=0.5, color='r')

    plt.grid(b=True, which='major', axis='both',

             linestyle="-", color="white")

    plt.legend(loc='upper right', numpoints=1, fontsize=12)

    plt.show()

    plt.clf()



def plot_two_fields(data, s, f1, f2,

                    pa, x_label, y_label,

                    axis_type):

    # d (list of lists) -- data from "get_data" function

    # s (string) -- search string

    # f1 (string) -- database field 1

    # f2 (string) -- database field 2

    # pa (list) -- plot axis

    # x_label (string) -- x-axis label

    # y_label (string) -- y-axis label

    # fn (string) -- figure name

    x0_list = []  # Fully Paid or Current

    y0_list = []  # Fully Paid or Current

    x1_list = []  # Late or Charged Off

    y1_list = []  # Late or Charged Off

    features_list = data[0]

    target_list = data[1]

    for i in range(len(features_list)):

        x = features_list[i][features_dict[f1]]

        y = features_list[i][features_dict[f2]]

        if target_list[i] == 0:

            x0_list.append(x)

            y0_list.append(y)

        elif target_list[i] == 1:

            x1_list.append(x)

            y1_list.append(y)

        else:

            pass

    create_scatter_plot(

        x0_list, y0_list,

        x1_list, y1_list,

        "Loan title search term: " + s, pa,

        x_label, y_label,

        axis_type)
cc_data = get_data('credit card')
plot_two_fields(cc_data, 'credit card', 'loan_amnt', 'int_rate',

                [1e2, 1e5, 5.0, 30.0], 'loan amount', 'interest rate',

                'semilogx')
plot_two_fields(cc_data, 'credit card', 'annual_inc', 'int_rate',

                [1e3, 1e7, 5.0, 30.0], 'annual income', 'interest rate',

                'semilogx')
plot_two_fields(cc_data, 'credit card', 'annual_inc', 'loan_amnt',

                [1e3, 1e7, 0.0, 35000.0], 'annual income', 'loan amount',

                'semilogx')
plot_two_fields(cc_data, 'credit card', 'loan_amnt', 'funded_amnt',

                [0.0, 35000.0, 0.0, 35000.0], 'loan amount', 'funded amount',

                'standard')
plot_two_fields(cc_data, 'credit card', 'home_ownership', 'funded_amnt',

                [-1, 6, 0.0, 35000.0], 'home ownership', 'funded amount',

                'standard')
medical_data = get_data('medical')
plot_two_fields(medical_data, 'medical', 'loan_amnt', 'int_rate',

                [1e2, 1e5, 5.0, 30.0], 'loan amount', 'interest rate',

                'semilogx')
plot_two_fields(medical_data, 'medical', 'annual_inc', 'int_rate',

                [1e3, 1e7, 5.0, 30.0], 'annual income', 'interest rate',

                'semilogx')
plot_two_fields(medical_data, 'medical', 'annual_inc', 'loan_amnt',

                [1e3, 1e7, 0.0, 35000.0], 'annual income', 'loan amount',

                'semilogx')
plot_two_fields(medical_data, 'medical', 'loan_amnt', 'funded_amnt',

                [0.0, 35000.0, 0.0, 35000.0], 'loan amount', 'funded amount',

                'standard')
plot_two_fields(medical_data, 'medical', 'home_ownership', 'funded_amnt',

                [-1, 6, 0.0, 35000.0], 'home ownership', 'funded amount',

                'standard')
debt_data = get_data('debt')
plot_two_fields(debt_data, 'debt', 'loan_amnt', 'int_rate',

                [1e2, 1e5, 5.0, 30.0], 'loan amount', 'interest rate',

                'semilogx')
plot_two_fields(debt_data, 'debt', 'annual_inc', 'int_rate',

                [1e3, 1e7, 5.0, 30.0], 'annual income', 'interest rate',

                'semilogx')
plot_two_fields(debt_data, 'debt', 'annual_inc', 'loan_amnt',

                [1e3, 1e7, 0.0, 35000.0], 'annual income', 'loan amount',

                'semilogx')
plot_two_fields(debt_data, 'debt', 'loan_amnt', 'funded_amnt',

                [0.0, 35000.0, 0.0, 35000.0], 'loan amount', 'funded amount',

                'standard')
plot_two_fields(debt_data, 'debt', 'home_ownership', 'funded_amnt',

                [-1, 6, 0.0, 35000.0], 'home ownership', 'funded amount',

                'standard')
def create_classifier(f, t, nt):

    """Create classifier for predicting loan status. Print accuracy.



    Arguments:

    f (list of tuples) -- [(sample 1 features), (sample 2 features),...]

    t (list)           -- [sample 1 target, sample 2 target,...]

    nt (int)           -- number of samples to use in training set

    """

    training_set_features = [] 

    training_set_target = [] 

    testing_set_features = []

    testing_set_target = []

    print("Number of training set samples:\t{0}".format(nt))

    print("Number of testing set samples:\t{0}".format(len(f)-nt))

    print("")

    # Build training set

    for i in np.arange(0, nt, 1):

        training_set_features.append(f[i])

        training_set_target.append(t[i])

    # Build testing set

    for i in np.arange(nt, len(f), 1):

        testing_set_features.append(f[i])

        testing_set_target.append(t[i])

    clf = tree.DecisionTreeClassifier()

    clf = clf.fit(training_set_features, training_set_target)

    n = 0

    n_correct = 0

    n0 = 0

    n0_correct = 0

    n1 = 0

    n1_correct = 0

    # Compare predictions to testing data

    for i in range(len(testing_set_features)):

        t = testing_set_target[i]

        p = clf.predict(np.asarray(testing_set_features[i]).reshape(1, -1))

        # Category 0

        if t == 0:

            if t == p[0]:

                equal = "yes"

                n_correct += 1

                n0_correct += 1

            else:

                equal = "no"

            n += 1

            n0 += 1

        # Category 1

        elif t == 1:

            if t == p[0]:

                equal = "yes"

                n_correct += 1

                n1_correct += 1

            else:

                equal = "no"

            n += 1

            n1 += 1

        else:

            pass

    n_accuracy = 100.0 * n_correct / n

    n0_accuracy = 100.0 * n0_correct / n0

    n1_accuracy = 100.0 * n1_correct / n1

    print("Accuracy of predicting testing set target values:")

    # Accuracy - manual calculation:

    print("    All samples (method 1): {0:3.4f}%".format(n_accuracy))

    # Accuracy - scikit-learn built-in method:

    print("    All samples (method 2): {0:3.4f}%".format(

          100.0 * clf.score(testing_set_features, testing_set_target)))

    print("    Samples with target=0: {0:3.4f}%".format(n0_accuracy))

    print("    Samples with target=1: {0:3.4f}%\n".format(n1_accuracy))
create_classifier(cc_data[0], cc_data[1], 2000)
create_classifier(medical_data[0], medical_data[1], 2000)
create_classifier(debt_data[0], debt_data[1], 2000)