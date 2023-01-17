import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import csv

solution_file = open('mySolution.csv', 'w', encoding='utf-8')

solution_writer = csv.writer(solution_file)

solution_writer.writerow(["Id","Category"])
import random

with open( '/kaggle/input/hanwhasystemict2020fest/heartbeat_question.csv', 'r' ) as theFile:

    reader = csv.DictReader(theFile)

    targetdata = [line for line in reader]  

    for x in targetdata:

        temp = random.randint(1, 5)

        if temp < 1:

            category='N'

        elif temp == 2:

            category='S'

        elif temp == 3:

            category='V'

        elif temp == 4:

            category='F'

        else:

            category='Q'

            

        solution_writer.writerow([x['Id'], category])



solution_file.close()



    