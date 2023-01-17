# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting
%matplotlib inline

import re #regex
from difflib import SequenceMatcher as SM #string comparison

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
cwur = pd.read_csv('../input/cwurData.csv')
shanghai = pd.read_csv('../input/shanghaiData.csv')
times = pd.read_csv('../input/timesData.csv')
cwur = cwur.rename(columns = {'institution': 'university_name'})
shanghai = shanghai.rename(columns = {'total_score': 'score'})
times = times.rename(columns = {'total_score': 'score'})
cwur = cwur[['university_name', 'score']]
shanghai = shanghai[['university_name', 'score']]
times = times[['university_name', 'score']]
times = times[~(times['score'] == '-')]
cwur.score = pd.to_numeric(cwur.score, errors='coerce')
shanghai.score = pd.to_numeric(shanghai.score, errors='coerce')
times.score = pd.to_numeric(shanghai.score, errors='coerce')
cwur = cwur.groupby('university_name').mean().reset_index()
shanghai = shanghai.groupby('university_name').mean().reset_index()
times = times.groupby('university_name').mean().reset_index()
scores_cwur = [x for x in cwur['score'].values if not np.isnan(x)]
scores_shanghai = [x for x in shanghai['score'].values if not np.isnan(x)]
scores_times = [x for x in times['score'].values if not np.isnan(x)]
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))

num_bins = 20
n, bins, patches = axes[0].hist(scores_cwur, num_bins, 
                                facecolor='green', alpha=0.5)
axes[0].set_title('Cwur')
n, bins, patches = axes[1].hist(scores_shanghai, num_bins,
                                facecolor='blue', alpha=0.5)
axes[1].set_title('Shanghai')
n, bins, patches = axes[2].hist(scores_times, num_bins, 
                                facecolor='red', alpha=0.5)
axes[2].set_title('Times')
plt.show()
boxplt = plt.boxplot([scores_cwur, scores_shanghai, scores_times])
labels = plt.xticks([1, 2, 3], ['Cwur', 'Shanghai', 'Times'])
cwur.score = (cwur.score - cwur.score.mean())/cwur.score.std()
shanghai.score = (shanghai.score - shanghai.score.mean())/shanghai.score.std()
times.score = (times.score - times.score.mean())/times.score.std()
z_scores_cwur = [x for x in cwur['score'].values if not np.isnan(x)]
z_scores_shanghai = [x for x in shanghai['score'].values if not np.isnan(x)]
z_scores_times = [x for x in times['score'].values if not np.isnan(x)]
boxplt = plt.boxplot([z_scores_cwur, z_scores_shanghai, z_scores_times])
labels = plt.xticks([1, 2, 3], ['Cwur', 'Shanghai', 'Times'])
cwur.university_name = [re.sub(r'[^a-zA-Z\s0-9]+', '', string) for string in cwur.university_name]
shanghai.university_name = [re.sub(r'[^a-zA-Z\s0-9]+', '', string) for string in shanghai.university_name]
times.university_name = [re.sub(r'[^a-zA-Z\s0-9]+', '', string) for string in times.university_name]
def is_fuzzy_match(string1, string2, threshold = 0.9):
    similarity = SM(None, str(string1), str(string2)).ratio()
    if (similarity > threshold):
        return True
    else:
        return False
(len(cwur), len(shanghai), len(times))
data = cwur
data = data.rename(columns = {'score': 'score_cwur'})
def check_for_uni_shanghai(series):
    university = series['university_name']
    for uni in shanghai['university_name'].values:
        if (is_fuzzy_match(university, uni)):
            return shanghai[shanghai['university_name'] == uni]['score'].values[0]
    #print('found no match for {u}'.format(u = university))
def check_for_uni_times(series):
    university = series['university_name']
    for uni in times['university_name'].values:
        if (is_fuzzy_match(university, uni)):
            return times[times['university_name'] == uni]['score'].values[0]
    #print('found no match for {u}'.format(u = university))
data['score_shanghai'] = data.apply(check_for_uni_shanghai, axis = 1)
data['score_times'] = data.apply(check_for_uni_times, axis = 1)
data[data.university_name == 'Harvard University']
(cwur[cwur.university_name == 'Harvard University'].score.values[0],
 shanghai[shanghai.university_name == 'Harvard University'].score.values[0],
 times[times.university_name == 'Harvard University'].score.values[0])
def calcScore(series):
    scores = [x for x in series.values[1:] if not np.isnan(x)]
    return np.mean(scores)
data['mean_score'] = data.apply(calcScore, axis = 1)
data = data.sort_values('mean_score', ascending=False)
data.to_csv('aggregatedScores.csv', sep=',')
programs = ['Auburn University', 'The University of Alabama',
       'Arkansas Tech University', 'University of Arkansas',
       'University of Arkansas at Little Rock', 'Arizona State University',
       'University of Arizona', 'California Polytechnic State University',
       'California State UniversityEast Bay',
       'California State UniversityFullerton',
       'California State UniversityLong Beach',
       'California State UniversitySan Bernardino', 'Chapman University',
       'Claremont Graduate University', 'Galvanize U',
       'National University', 'San Jose State University',
       'Santa Clara University', 'Stanford University',
       'University of California Hastings College of Law',
       'University of CaliforniaDavis', 'University of CaliforniaIrvine',
       'University of CaliforniaSan Diego',
       'University of CaliforniaBerkeley', 'University of San Francisco',
       'University Of San Francisco', 'University of Southern California',
       'University of the Pacific', 'American Sentinel University',
       'Aspen University', 'Colorado State UniversityFort Collins',
       'Colorado State UniversityGlobal Campus',
       'Colorado Technical University', 'Regis University',
       'University of Colorado Boulder', 'University of Colorado Denver',
       'University of Denver', 'Central Connecticut State University',
       'Quinnipiac University', 'University of Connecticut',
       'University of New Haven', 'American University',
       'George Washington University', 'Georgetown University',
       'The George Washington University', 'New College of Florida',
       'Florida International University',
       'Florida Polytechnic University', 'Full Sail University',
       'Nova Southeastern University', 'PascoHernando State College',
       'University of Central Florida', 'University of Florida',
       'University of Miami',
       'University of South Florida SarasotaManatee',
       'University of South FloridaMain Campus',
       'Georgia Southern University', 'Georgia State University',
       'Georgia Tech', 'Kennesaw State University', 'Mercer University',
       'University of Georgia', 'Loras College', 'Northwestern College',
       'Luther College', 'The University of Iowa', 'Aurora University',
       'Benedictine University', 'DePaul University', 'Elmhurst College',
       'Illinois Institute of Technology', 'Lewis University',
       'Loyola University Chicago', 'Northwestern University',
       'University of Chicago', 'University of Illinois at Chicago',
       'University of Illinois at Springfield',
       'University of Illinois at UrbanaChampaign',
       'University of St Francis', 'Indiana University Bloomington',
       'Indiana UniversityPurdue UniversityIndianapolis',
       'Purdue UniversityMain Campus', 'Saint Marys College',
       'University of Notre Dame', 'University of Evansville',
       'University of Kansas', 'Northern Kentucky University',
       'University of Louisville', 'Louisiana State University',
       'Babson College', 'Becker College', 'Bentley University',
       'Brandeis University', 'Harvard University',
       'Northeastern University', 'University of Massachusetts Amherst',
       'Worcester Polytechnic Institute', 'Smith College',
       'Johns Hopkins University', 'Notre Dame of Maryland University',
       'University of MarylandBaltimore County',
       'University of MarylandCollege Park',
       'University of MarylandUniversity College', 'Baker College',
       'Central Michigan University', 'Davenport University',
       'Eastern Michigan University', 'Grand Valley State University',
       'Michigan State University', 'Michigan Technological University',
       'Oakland University', 'University of MichiganAnn Arbor',
       'University of MichiganDearborn', 'Capella University',
       'The College of Saint Scholastica', 'University of Minnesota',
       'University of MinnesotaDuluth', 'University of St Thomas',
       'Winona State University', 'Grantham University',
       'Missouri University of Science and Technology',
       'Rockhurst University', 'Saint Louis University',
       'Saint Louis UniversityMain Campus',
       'University of MissouriSt Louis', 'Jackson State University',
       'University of Montana', 'Elon University',
       'North Carolina State University at Raleigh',
       'University of North Carolina at Chapel Hill',
       'University of North Carolina at Charlotte',
       'University of North Carolina at Greensboro',
       'Wake forest University', 'Bellevue University',
       'Creighton University', 'Nebraska College of Technical Agriculture',
       'University of Nebraska at Omaha',
       'Southern New Hampshire University',
       'New Jersey Institute of Technology', 'Rutgers University',
       'Saint Peters University', 'Stevens Institute of Technology',
       'Thomas Edison State College', 'University of NevadaReno',
       'Columbia University in the City of New York', 'Cornell University',
       'CUNY Bernard M Baruch College',
       'CUNY Graduate School and University Center', 'CUNY Queens College',
       'Fordham University', 'Keller Graduate School of Management',
       'Marist College', 'New York University', 'Pace UniversityNew York',
       'Rensselaer Polytechnic Institute', 'St Johns UniversityNew York',
       'Syracuse University', 'The New School', 'Trocaire College',
       'Union Graduate College', 'University at Buffalo',
       'University of Rochester', 'Bowling Green State University',
       'Case Western Reserve University', 'Cleveland State University',
       'Miami University of Ohio', 'Notre Dame College', 'Ohio University',
       'The Ohio State University', 'University of CincinnatiMain Campus',
       'Oklahoma State University Center for Health Sciences',
       'Southwestern Oklahoma State University',
       'University of Oklahoma Norman Campus', 'Oregon State University',
       'Albright College', 'Carnegie Mellon University',
       'Drexel University',
       'Harrisburg University of Science and Technology',
       'La Salle University', 'Misericordia University',
       'Pennsylvania State University', 'Philadelphia University',
       'Saint Josephs University', 'Temple University',
       'University of PittsburghBradford',
       'University of PittsburghPittsburgh Campus', 'Villanova University',
       'Brown University', 'College of Charleston',
       'Medical University of South Carolina',
       'University of South CarolinaColumbia', 'Dakota State University',
       'South Dakota State University', 'Austin Peay State University',
       'Middle Tennessee State University',
       'Tennessee Technological University', 'The University of Tennessee',
       'The University of Tennessee at Chattanooga',
       'University of Memphis', 'Southern Methodist University',
       'St Marys University', 'Tarleton State University',
       'Texas A  M UniversityCollege Station',
       'The University of Texas at Austin',
       'The University of Texas at Dallas',
       'The University of Texas at San Antonio', 'University of Dallas',
       'University of North Texas', 'University of Utah',
       'George Mason University', 'Radford University',
       'University of Virginia', 'Virginia Commonwealth University',
       'Virginia Polytechnic Institute and State University',
       'Statisticscom', 'Bellevue College', 'City University of Seattle',
       'Seattle University', 'University of WashingtonSeattle Campus',
       'University of WashingtonTacoma Campus',
       'University of WisconsinMadison',
       'University of Wisconsin Colleges',
       'University of WisconsinMilwaukee', 'West Virginia University',
       'Ukrainian Catholic Univeristy', 'Sabanc University',
       'National University of Singapore', 'Dalarna University',
       'Blekinge Institute of Technology',
       'Kth Royal Institute Of Technology', 'Linkping University',
       'Universidade Nova de Lisboa', 'University of Otago',
       'Massey University', 'Erasmus University', 'Maastricht University',
       'Radboud Universiteit Nijmegen',
       'Eindhoven University of TechnologyTUe', 'Utrecht University',
       'Vrije Universiteit Amsterdam',
       'Autonomous Technological Institute of Mexico',
       'Mykolas Romeris University', 'Sangmyung University', 'BAICR',
       'Polytechnic University Of Turin', 'University Of MilanBicocca',
       'University Of Pisa', 'BenGurion University Of The Negev',
       'Dublin City University', 'Dublin Institute Of Technology',
       'Institute Of Technology Blanchardstown',
       'Irish Management Institute', 'National College Of Ireland',
       'National University Of Ireland Galway', 'University College Cork',
       'University College Dublin', 'Chinese University of Hong Kong',
       'Hong Kong University of Science  Technology',
       'Lancaster University', 'Aston University',
       'Birmingham City University', 'Bournemouth University',
       'Brunel University London', 'City University London',
       'Coventry University', 'De Montfort University',
       'Goldsmiths University of London', 'Imperial College London',
       'Leeds Met', 'Newcastle University', 'Robert Gordon University',
       'Royal Holloway University Of London',
       'Sheffield Hallam University', 'The University Of Edinburgh',
       'The University Of Manchester', 'University College London',
       'University Of Bristol', 'University of Derby',
       'University of Dundee', 'University Of East Anglia',
       'University Of East London', 'University Of Essex',
       'University Of Greenwich', 'University of Kent',
       'University Of Leeds', 'University of Leicester',
       'University Of Liverpool', 'University of Manchester',
       'University of Nottingham', 'University of Southampton',
       'University Of St Andrews', 'University of Strathclyde',
       'University of Surrey', 'University of Warwick',
       'University Of Warwick', 'University Of Westminster',
       'Data ScienceTech Institute', 'EISTI', 'ENSAE Paris Tech',
       'Telecom Paris Tech', 'Telecom Sudparis',
       'Universit Pierre Et Marie Curie', 'Aalto University',
       'University Of Helsinki', 'Universit De Nantes',
       'Barcelona School of Management', 'Instituto de Empresa',
       'Universidad Rey Juan Carlos', 'Universitat Pompeu Fabra',
       'Universities Of Alicante', 'University of Barcelona',
       'University of Oviedo', 'Aalborg University', 'Aarhus University',
       'Technical University of Denmark',
       'Otto Von Guericke University Magdeburg', 'TU Dortmund',
       'Universitt Konstanz', 'Queens University',
       'Simon Fraser University', 'University Of Alberta',
       'University of the Fraser Valley', 'York University',
       'Mackenzie Presbyterian Institute', 'Deakin University',
       'Macquarie University', 'University of South Australia',
       'University of Technology Sydney',
       'Vienna University of Economics and Business',
       'University of Vienna']
programs = [re.sub(r'[^a-zA-Z\s0-9]+', '', string) for string in programs]
def check_for_master(university):
    for program in programs:
        similarity = is_fuzzy_match(university, program)
        if (similarity):
            return True
data['ds_master'] = data['university_name'].apply(check_for_master)
universities_with_ds_program = data[data['ds_master'] == True]
universities_with_ds_program.head(n=20)
universities_with_ds_program.to_csv('DSDegrees.csv', sep = ',')
