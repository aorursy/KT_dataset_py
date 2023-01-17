import pandas as pd
sub = pd.read_csv('../input/titanic/gender_submission.csv')

sub.head()
sub['Survived'] = 0
sub.to_csv('submission.csv', index=False)