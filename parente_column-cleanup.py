import pandas as pd
df = pd.read_csv('../input/Survey.csv')
new_column_names = [
    'time_started',
    'date_submitted',
    'status',
    'how_often',
    'hinderances',
    'how_long_used',
    'integrations_1',
    'integrations_2',
    'integrations_3',
    'how_run',
    'how_run_other',
    'workflow_needs_addressed_1',
    'workflow_needs_addressed_2',
    'workflow_needs_addressed_3',
    'workflow_needs_not_addressed_1',
    'workflow_needs_not_addressed_2',
    'workflow_needs_not_addressed_3',
    'pleasant_aspects_1',
    'pleasant_aspects_2',
    'pleasant_aspects_3',
    'difficult_aspects_1',
    'difficult_aspects_2',
    'difficult_aspects_3',
    'features_changes_1',
    'features_changes_2',
    'features_changes_3',
    'first_experience_enhancements_1',
    'first_experience_enhancements_2',
    'first_experience_enhancements_3',
    'keywords',
    'keywords_other',
    'role',
    'years_in_role',
    'industry_1',
    'industry_2',
    'industry_3',
    'audience_size'
]
len(new_column_names)
df.columns = new_column_names
df.head()
df.to_csv('survey_short_columns.csv')