!pip3 install git+https://download.radtorch.com/ -q
from radtorch import pipeline, core

from radtorch.settings import *
data_dir = '/kaggle/input/rsna-str-pulmonary-embolism-detection/train/'

data_csv = '/kaggle/input/rsna-str-pulmonary-embolism-detection/train.csv'
df = pd.read_csv(data_csv)

df.head()
print ('EXTREME EXPLORATORY DATA ANALYSIS')

print('===================================')

print ('Number of Studies =', len(df.StudyInstanceUID.unique()))

print ('Number of Series =', len(df.SeriesInstanceUID.unique()))

print ('Number of Images =', len(df.SOPInstanceUID.unique()))

print ('Number of Studies with Positive PE =', len((df[df['negative_exam_for_pe']==0]).SeriesInstanceUID.unique()))  

print ('Number of Studies with Negative PE =', len((df[df['negative_exam_for_pe']==1]).SeriesInstanceUID.unique()))  

print ('Number of Images with positive PE within Positive Studies =', len(df.loc[(df['pe_present_on_image'] == 1) & (df['negative_exam_for_pe'] == 0)]))

print ('Number of Images with negative PE within Positive Studies =', len(df.loc[(df['pe_present_on_image'] == 0) & (df['negative_exam_for_pe'] == 0)]))

print ('Number of Images with positive PE within Negative Studies =', len(df.loc[(df['pe_present_on_image'] == 1) & (df['negative_exam_for_pe'] == 1)]))

print ('Number of Images with negative PE within Negative Studies =', len(df.loc[(df['pe_present_on_image'] == 0) & (df['negative_exam_for_pe'] == 1)]))

print ('')

print ('Number of Images with Right sided PE ONLY =', len(df.loc[(df['pe_present_on_image'] == 1) & (df['negative_exam_for_pe'] == 0)& (df['rightsided_pe'] == 1)& (df['leftsided_pe'] ==0)& (df['central_pe'] == 0)]))

print ('Number of Images with Left sided PE ONLY =', len(df.loc[(df['pe_present_on_image'] == 1) & (df['negative_exam_for_pe'] == 0)& (df['leftsided_pe'] == 1)& (df['rightsided_pe'] == 0)&(df['central_pe'] == 0)]))

print ('Number of Images with Central PE ONLY =', len(df.loc[(df['pe_present_on_image'] == 1) & (df['negative_exam_for_pe'] == 0)& (df['central_pe'] == 1)&(df['rightsided_pe'] == 0)& (df['leftsided_pe'] == 0)]))

print ('')

print ('Number of Images with Right & Left PE =', len(df.loc[(df['pe_present_on_image'] == 1) & (df['negative_exam_for_pe'] == 0)& (df['rightsided_pe'] == 1)& (df['leftsided_pe'] == 1)]))

print ('Number of Images with Right & Central PE =', len(df.loc[(df['pe_present_on_image'] == 1) & (df['negative_exam_for_pe'] == 0)& (df['rightsided_pe'] == 1)& (df['central_pe'] == 1)]))

print ('Number of Images with Left & Central PE =', len(df.loc[(df['pe_present_on_image'] == 1) & (df['negative_exam_for_pe'] == 0)& (df['leftsided_pe'] == 1)& (df['central_pe'] == 1)]))

print ('Number of Images with Right/Left/Central PE =', len(df.loc[(df['pe_present_on_image'] == 1) & (df['negative_exam_for_pe'] == 0)& (df['rightsided_pe'] == 1)& (df['leftsided_pe'] == 1)& (df['central_pe'] == 1)]))

print ('')

print ('Number of Studies which are indeterminate for PE =', len(df[df['indeterminate']==1]))

print ('Number of Studies which are indeterminate b/c contrast issues ONLY =', len(df.loc[(df['indeterminate'] == 1) & (df['qa_contrast'] == 1)& (df['qa_motion'] == 0)]))

print ('Number of Studies which are indeterminate b/c motion issues ONLY =', len(df.loc[(df['indeterminate'] == 1) & (df['qa_motion'] == 1)& (df['qa_contrast'] == 0)]))

print ('Number of Studies which are indeterminate b/c contrast and motion issues =', len(df.loc[(df['indeterminate'] == 1) & (df['qa_motion'] == 1)& (df['qa_contrast'] == 1)]))

print ('')

chronic = len(df.loc[(df['pe_present_on_image'] == 1) & (df['chronic_pe'] == 1)])

acute_chronic = len(df.loc[(df['pe_present_on_image'] == 1) & (df['acute_and_chronic_pe'] == 1)])

acute = len(df.loc[(df['pe_present_on_image'] == 1) & (df['negative_exam_for_pe'] == 0)])-(chronic+acute_chronic)

print ('Number of Images with positive PE and Acute =', acute)

print ('Number of Images with positive PE and Chronic =', chronic)

print ('Number of Images with positive PE and Acute/Chronic =', acute_chronic)

print ('')

print ('Number of Images with positive PE with flow artifact =', len(df.loc[(df['pe_present_on_image'] == 1) & (df['flow_artifact'] == 1)]))

print ('Number of Images with positive PE without flow artifact =', len(df.loc[(df['pe_present_on_image'] == 1) & (df['flow_artifact'] == 0)]))

print ('Number of Images with negative PE with flow artifact =', len(df.loc[(df['pe_present_on_image'] == 0) & (df['flow_artifact'] == 1)]))

print ('Number of Images with negative PE without flow artifact =', len(df.loc[(df['pe_present_on_image'] == 0) & (df['flow_artifact'] == 0)]))

print ('')

print ('Number of Images with positive PE with true_filling_defect_not_pe =', len(df.loc[(df['pe_present_on_image'] == 1) & (df['true_filling_defect_not_pe'] == 1)]))

print ('Number of Images with positive PE without true_filling_defect_not_pe =', len(df.loc[(df['pe_present_on_image'] == 1) & (df['true_filling_defect_not_pe'] == 0)]))

print ('Number of Images with negative PE with true_filling_defect_not_pe =', len(df.loc[(df['pe_present_on_image'] == 0) & (df['true_filling_defect_not_pe'] == 1)&(df['indeterminate'] == 0)]))

print ('Number of Images with negative PE without true_filling_defect_not_pe =', len(df.loc[(df['pe_present_on_image'] == 0) & (df['true_filling_defect_not_pe'] == 0)&(df['indeterminate'] == 0)]))
