import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")

%pylab inline



from subprocess import check_output

print(check_output(["ls", '-R', "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
'''with open('../input/solutions.csv', 'r') as fl:

    lines = fl.readlines()

newlines = []

for index, line in enumerate(lines):

    index += 1

    if index == 326311:

        line1 = line[:54].strip()

        line2 = line[54:].strip()

        newlines.append(line1+'\n')

        newlines.append(line2+'\n')

    else:

        newlines.append(line)

with open('solutions.csv', 'w') as fl:

    fl.writelines(newlines)

solutions = pd.read_csv('solutions.csv')'''
#print(solutions.shape)

#solutions = solutions.drop_duplicates()

#print(solutions.shape)

#solutions.to_csv('solutions.csv', index=False)
def load_nodup(name):

    df = pd.read_csv('../input/program_codes/'+name+'.csv')

    df = df.drop_duplicates(subset='Solutions')

    return df

#f = load_nodup('first')

#f.to_csv('first.csv', index=False)
#s = load_nodup('second')

#s.to_csv('second.csv', index=False)
t = load_nodup('third')

t.to_csv('third.csv', index=False)