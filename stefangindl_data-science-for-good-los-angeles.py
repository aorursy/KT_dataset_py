!pip install https://github.com/andreasvc/readability/tarball/master
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
import csv
import hashlib
import re

from more_itertools import flatten
from nltk import word_tokenize
from toolz.itertoolz import partition, sliding_window
import doctest
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import readability
import seaborn as sns


nltk.download('punkt')
csv_path = 'structured_bulletins.csv'
# http://www.act.org/content/act/en/research/reports/act-publications/college-choice-report-class-of-2013/college-majors-and-occupational-choices/college-majors-and-occupational-choices.html
LIST_OF_MAJORS = [
 'architecture',
 'area studies',
 'art',
 'art education',
 'art history',
 'asian area studies',
 'asian languages & literatures',
 'astronomy',
 'athletic training',
 'atmospheric sciences & meteorology',
 'autobody repair/technology',
 'automotive engineering technology',
 'automotive mechanics/technology',
 'aviation & airway science',
 'aviation management & operations',
 'avionics technology',
 'banking & financial support services',
 'bible studies',
 'biblical studies',
 'biochemistry & biophysics',
 'bioengineering ',
 'biology',
 'biomedical engineering',
 'biomedical engineering technologies',
 'broadcast',
 'building technology',
 'business administration & management',
 'business economics',
 'business education',
 'business quantitative methods',
 'cad technology',
 'career & technical education',
 'carpentry',
 'cell biology',
 'cellular biology',
 'chemical engineering',
 'chemistry',
 'chicano studies',
 'child care services management',
 'child development',
 'chiropractic (pre-chiropractic)',
 'cinema ',
 'cinematography production',
 'city planning',
 'civil engineering',
 'civil engineering technology',
 'classical languages & literatures',
 'clinical & counseling',
 'clinical assisting',
 'communication disorder services (e.g.',
 'communications',
 'communications technology',
 'community organization & advocacy',
 'comparative literature',
 'computer & information sciences',
 'computer engineering',
 'computer engineering technology',
 'computer networking/telecommunications',
 'computer science & programming',
 'computer software & media applications',
 'computer system administration',
 'construction engineering/management',
 'construction technology',
 'construction trades (e.g.',
 'consumer & family economics',
 'contracts management',
 'corrections',
 'cosmetology ',
 'counseling & student services',
 'court reporting',
 'creative writing',
 'criminal justice',
 'criminology',
 'criticism & conservation',
 'culinary arts/chef training',
 'curriculum & instruction',
 'dance',
 'data management technology',
 'dental assisting',
 'dental hygiene',
 'dentistry (pre-dentistry)',
 'design & visual communications',
 'diesel mechanics/technology',
 'digital communications/media',
 'divinity ',
 'drafting technology',
 'drug abuse counseling',
 'early childhood education',
 'ecology',
 'economics',
 'educational administration',
 'electrical',
 'electrical',
 'electrical equip installation & repair',
 'electrical)',
 'electromechanical engineering technologies',
 'electronics & communications engineering',
 'electronics engineering technologies',
 'electronics equip installation & repair',
 'elementary education',
 'emergency medical technology',
 'engineering (pre-engineering)',
 'engineering technology',
 'english arts education',
 'english language & literature',
 'english literature',
 'english-as-a-second-language education',
 'environmental control technologies',
 'environmental health engineering',
 'environmental science',
 'ethnic & minority studies',
 'european area studies',
 'exercise science/physiology/kinesiology',
 'facilities administration',
 'family & consumer sciences',
 'fashion design',
 'fashion merchandising',
 'film ',
 'film production',
 'finance',
 'financial planning & services',
 'fine arts',
 'fire protection & safety technology',
 'food & nutrition',
 'food sciences & technology',
 'food services management',
 'foreign languages education',
 'foreign languages/literatures',
 'forestry',
 'french language & literature',
 'funeral services & mortuary science',
 'general',
 'genetics',
 'geography',
 'geological & earth sciences',
 'german language & literature',
 'graphic & printing equipment operation',
 'graphic design',
 'ground',
 'hairstyling ',
 'health & physical education/fitness',
 'health education',
 'health services administration',
 'health technology',
 'health-related professions & services',
 'heating cond/refrig install/repair',
 'history',
 'horticulture operations & management',
 'horticulture science',
 'hospital administration',
 'hotel management',
 'human resources development/training',
 'human resources management',
 'industrial design',
 'industrial engineering',
 'industrial production technologies',
 'industrial relations',
 'information science',
 'insurance & risk management',
 'interdisciplinary studies',
 'interior architecture',
 'interior design',
 'international business management',
 'international relations & affairs',
 'investments & securities',
 'journalism',
 'journalism',
 'junior high/middle school education',
 'labor relations',
 'landscape architecture',
 'language arts education',
 'latin american area studies',
 'latino studies',
 'law (pre-law)',
 'law enforcement',
 'legal administrative assisting/secretarial',
 'legal assistant',
 'legal studies',
 'leisure facilities management',
 'liberal arts & general studies',
 'library science',
 'linguistics',
 'logistics & materials management',
 'machine tool technology',
 'management information systems',
 'management quantitative methods',
 'managerial economics',
 'marine biology',
 'marketing management & research',
 'mass communications',
 'massage therapy',
 'mathematics',
 'mathematics education',
 'mechanical drafting/cad technology',
 'mechanical engineering',
 'mechanical engineering technology',
 'mechanics & repairers',
 'medical assisting',
 'medical assisting',
 'medical laboratory technology',
 'medical office/secretarial',
 'medical radiologic technology',
 'medical records',
 'medical technology',
 'medicine (pre-medicine)',
 'mental health counseling',
 'mental health technician',
 'merchandising',
 'microbiology & immunology',
 'middle eastern languages & literatures',
 'middle eastern)',
 'military technologies',
 'ministry ',
 'motel management',
 'multi studies',
 'multimedia effects',
 'music',
 'music',
 'music',
 'music education',
 'natural resources conservation',
 'natural resources management',
 'north american area studies',
 'nuclear engineering',
 'nuclear medicine technology',
 'nursing',
 'nursing',
 'occupational therapy',
 'occupational therapy assisting',
 'office supervision & management',
 'operations management & supervision',
 'optometry (pre-optometry)',
 'organizational behavior',
 'osteopathic medicine',
 'paralegal assistant',
 'parks',
 'parks facilities management',
 'performance',
 'personal services',
 'pharmacy (pre-pharmacy)',
 'philosophy',
 'photography',
 'physical education & coaching',
 'physical sciences',
 'physical therapy (pre-physical therapy)',
 'physical therapy assisting',
 'physician assisting',
 'physics',
 'plumbing',
 'political science & government',
 'postsecondary education',
 'practical (lpn)',
 'precision production trades',
 'print',
 'procurement management',
 'protective services',
 'psychiatric health technician',
 'psychology',
 'psychology',
 'public administration',
 'public administration & services',
 'public affairs & public policy analysis',
 'public health',
 'public relations & organizational communication',
 'public speaking',
 'purchasing management',
 'quality control & safety technologies',
 'radio & television broadcasting',
 'radio & television broadcasting technology',
 'real estate',
 'rec facilities management',
 'recreation',
 'regional planning',
 'registered (bs/rn)',
 'rehabilitation therapy',
 'religion',
 'religious education',
 'respiratory therapy technology',
 'restaurant services management',
 'sales',
 'science education',
 'secondary education',
 'secretarial studies & office administration',
 'small business management/operations',
 'social sciences',
 'social studies/sciences education',
 'social work',
 'sociology',
 'spanish language & literature',
 'special education',
 'special effects',
 'speech pathology)',
 'sport & fitness administration/management',
 'statistics',
 'studio arts',
 'subject-specific',
 'surgical technology',
 'surveying technology',
 'teacher assisting/aide education',
 'teacher education',
 'teacher education',
 'textile & apparel',
 'theatre arts/drama',
 'theology',
 'theory & composition',
 'therapy & rehabilitation',
 'tourism & travel marketing',
 'tourism management',
 'transportation & materials moving (e.g.',
 'travel management',
 'urban planning',
 'urban studies/urban affairs',
 'veterinarian assisting/technology',
 'veterinary medicine (pre-veterinarian)',
 'vide production',
 'vocational (lpn)',
 'vocational rehabilitation counseling',
 'webpage design',
 'welding technology',
 'wildlife & wildlands management',
 'women’s studies',
 'zoology']
FOLDER_JOB_BULLETINS = Path("../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins")
REGEXES = {'ANNUAL SALARY': re.compile('ANNUAL ?SALARY'),
           'DUTIES': re.compile('DUTIES'),
           'REQS': re.compile('((REQUIREMENTS?)|(/MINIMUM QUALIFICATIONS))+'),
           'WHERE TO APPLY': re.compile('(HOW|WHERE) TO APPLY'),
           'APPLICATION DEADLINE': re.compile('APPLICATION DEADLINE'),
           'SELECTION PROCESS': re.compile('SELEL?CTION PROCE(SS|DURE)'),
          }
written_num_to_int = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 
                      'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                      'eleven': 11, 'twelve': 12}
time_length_as_year_fraction = {'day': 1/365, 'week': 1/52, 'month': 1/12, 'year': 1}

EDUCATION_OPTIONS = ['COLLEGE OR UNIVERSITY', 'HIGH SCHOOL', 'APPRENTICESHIP']
EMPLOYMENT_OPTIONS = ['full-time', 'part-time']


def get_salaries(salary):
    """
    Salaries having a range, i.e. $100.000 to $200.000:
    >>> salary = "\\n$64,665 to $94,502 and $83,373 to $121,897\\n\\nNOTES:\\n\\n1. Candidates from the eligible list are normally appointed to vacancies in the lower pay grade positions.\\n2. The current salary range is subject to change. You may confirm the starting salary with the hiring department before accepting a job offer.\\n"
    >>> get_salaries(salary)
    '$64,665 - $94,502'

    Salaries without a range, i.e. $100.000:
    >>> salary = '$82,350 (flat-rated) '
    >>> get_salaries(salary)
    '$82,350 (flat-rated)'
    """
    if not isinstance(salary, str):
        return np.NaN
    salaries = []
    raw_salaries = re.findall('\$ *[0-9]{1,3},[0-9]{3} \(flat-rated\)', salary)
    if len(raw_salaries) == 1:
        return raw_salaries[0]
    for salary_from, salary_to in list(partition(2, re.findall('\$ *[0-9]{1,3},[0-9]{3}', salary))):
        salaries.append(f'{salary_from.replace(" ", "")} - {salary_to.replace(" ", "")}')
    return np.NaN if len(salaries) == 0 else salaries[0]


def get_working_hours(requirements):
    """

    Only full-time option:
    >>> requirements = '\\nOne year of full-time paid experience with the City of Los Angeles as a Port Police Lieutenant or in a class at that level providing necessary law enforcement, safety, and police services.\\n'
    >>> get_working_hours(requirements)
    'full-time'

    Full-time and part-time option:
    >>> requirements = 'Graduation from an accredited four-year college or university with a major in a natural science or education, and one year of full-time or two years of part-time paid or volunteer out of classroom (informal) or in classroom (formal) teaching experience which includes marine or life science in the curriculum;'
    >>> get_working_hours(requirements)
    'full-time/part-time'
    """
    employment = [x for x in EMPLOYMENT_OPTIONS if x in requirements]
    return np.NaN if len(employment) == 0 else '/'.join(employment)

def get_education(requirements):
    """
    College or university:
    >>> requirements = '1. Two years of full-time paid professional experience as a Principal Public Relations Representative or in a class at that level with responsibility for public relations program management and development with the City of Los Angeles; or 2. Graduation from an accredited four-year college or university with a degree in journalism, English, public relations, or communications and two years of full-time paid professional experience in supervising a public relations staff responsible for a comprehensive public information program; or 3.Graduation from an accredited four-year college or university and four years of full-time paid professional experience in public relations managing a comprehensive public information program.'
    >>> get_education(requirements)
    'COLLEGE OR UNIVERSITY'

    High school, as the lower of the two "high school" and "college":
    >>> requirements = 'One year of high school or college level algebra'
    >>> get_education(requirements)
    'HIGH SCHOOL'
    """

    education_reqs = [x for x in EDUCATION_OPTIONS if x in requirements.upper()]
    return np.NaN if len(education_reqs) == 0 else '/'.join(education_reqs)


def convert_education_duration_to_num(duration):
    """
    >>> duration = ('two', 'year')
    >>> convert_education_duration_to_num(duration)
    2
    >>> duration = ('six', 'month')
    >>> convert_education_duration_to_num(duration)
    0.5
    >>> duration = ('five', 'day')
    >>> round(convert_education_duration_to_num(duration), 3)
    0.014
    """
    num = written_num_to_int[duration[0]]
    frac_of_year = time_length_as_year_fraction[duration[1]]
    return num * frac_of_year


def extract_education_times(in_str):
    """
    >>> in_str = '1. Graduation from an accredited four-year college or university with a major in Computer Science, Information Systems, or Geographical Information Systems; or'
    >>> extract_education_times(in_str)
    [('four', 'year')]
    
    >>> in_str = '3. Two years of full-time paid experience as a Systems Aide with the City of Los Angeles; and'
    >>> extract_education_times(in_str)
    [('two', 'year')]
    
    >>> in_str = '2. Graduation from an accredited four-year college or university and two years of full-time paid experience in a class at the level of Management Assistant which provides experience in:'
    >>> extract_education_times(in_str)
    [('four', 'year'), ('two', 'year')]
    
    """
    return re.findall('(one|two|three|four|five|six|seven|eight|nine)[- ]+(year|month|week|week|day)', in_str.lower())


def get_education_years(in_str):
    """
    >>> in_str = '1. Graduation from an accredited four-year college or university with a major in Computer Science, Information Systems, or Geographical Information Systems; or'
    >>> get_education_years(in_str)
    4
    >>> in_str = '2. Graduation from an accredited six-month college or university and two years of full-time paid experience in a class at the level of Management Assistant which provides experience in:'
    >>> get_education_years(in_str)
    2.5
    """
    if in_str is np.NaN:
        return in_str
    years = extract_education_times(in_str)
    if not years:
        return np.NaN
    years = [convert_education_duration_to_num(el) for el in years]
    years = sum(years)
    return years


def get_drivers_license_info(requirements):
    """
    Driver's license is required:
    >>> requirements = "A valid California driver's license is required. Applicants will be disqualified and not eligible for hire if their record within the last 36 months reflects three or more moving violations and/or at-fault accidents, or conviction of a major moving violation (such as DUI)."
    >>> get_drivers_license_info(requirements)
    'R'
    
    Driver's license may be required:
    >>> requirements = "Some positions may require a valid California driver's license. Candidates may not be eligible for appointment to these positions if their record within the last 36 months reflects three or more moving violations and/or at-fault accidents, or a conviction of a major moving violation (such as DUI)."
    >>> get_drivers_license_info(requirements)
    'P'
    """
    if "driver's license is required" in requirements:
        return 'R'
    if "may require a valid California driver's license" in requirements:
        return 'P'
    return np.NaN


def get_drivers_license_class(reqs):
    """
    
    Driver's license class B:
    >>> reqs = "A valid California Class B driver's license may be required prior to appointment."
    >>> get_drivers_license_class(reqs)
    'B'
    
    >>> reqs = 'A valid California Class "B" commercial driver\\'s license with ...'
    >>> get_drivers_license_class(reqs)
    'B'
    
    Class A or B:
    >>> reqs = "Some positions may require a valid California Class A or B driver's license"
    >>> get_drivers_license_class(reqs)
    'A, B'
    
    Class B or C:
    >>> reqs = "Some positions may require a valid California Class C or B driver's license prior to appointment."
    >>> get_drivers_license_class(reqs)
    'B, C'
    
    Class C:
    >>> reqs = "Some positions may require a valid California Class C driver's license;"
    >>> get_drivers_license_class(reqs)
    'C'
    
    >>> reqs = "Some positions may require a valid California Class B (or A) driver's license"
    >>> get_drivers_license_class(reqs)
    'A, B'
    """
    license_classes = {'Class A': 'A', 'Class B': 'B', 'Class C': 'C',
                       'Class "A"': 'A', 'Class "B"': 'B', 'Class "C"': 'C'}
    
    if not "driver's license" in reqs:
        return np.NaN
    
    licenses = set()
    licenses.update([abbrv for lic_class, abbrv in license_classes.items() if lic_class in reqs])
    
    m = re.findall("(?<=Class [A|B|C] \(or )[A|B|C]", reqs)
    if len(m) > 0:
        licenses.update(m[0])
        
    m = re.findall("(?<=Class [A|B|C] or )[A|B|C]", reqs)
    if len(m) > 0:
        licenses.update(m[0])
    
    return ', '.join(sorted(licenses))


def get_dwp_salary(salary):
    """
    
    >>> salary = "The salary in the Department of Water and Power is $93,542 (flat-rated)"
    >>> get_dwp_salary(salary)
    '$93,542 (flat-rated)'
    
    >>> salary = "The salary range for positions in the Department of Water and Power is $72,328 to $89,864"
    >>> get_dwp_salary(salary)
    '$72,328 - $89,864'
    
    >>> salary = "The salary in the Department of Water and Power is $70,908 to $88,092 and $83,770 to $104,065."
    >>> get_dwp_salary(salary)
    '$70,908 - $88,092'
    """
    if not isinstance(salary, str):
        return np.NaN
    
    m = re.search("(?<=The salary in the Department of Water and Power is ){1}\$[0-9]{1,3},[0-9]{3} \(flat-rated\)", salary)
    if m:
        return m[0]
    
    m = re.search("(?<=Department of Water and Power is ){1}\$[0-9]{1,3},[0-9]{3} to \$[0-9]{1,3},[0-9]{3}", salary)
    if m:
        return ' - '.join(m[0].split(" to "))
    
    return np.NaN


def get_additional_licenses(reqs):
    """
    
    Medical certificate (WELDER 3796 102816.txt)
    >>> reqs = "Some positions may require a valid California Class C driver's license; or a valid Class B driver's license and valid medical certificate approved by the State of California Department of Motor Vehicles, prior to appointment."
    >>> get_additional_licenses(reqs)
    'Medical Certificate'
    
    Medical certificate (WATER UTILITY WORKER 3912 120817.txt)
    >>> reqs = "Some positions may require a valid California Class B (or A) driver's license and valid medical certificate approved by the State of California Department of Motor Vehicles prior to appointment."
    >>> get_additional_licenses(reqs)
    'Medical Certificate'
    
    """
    if 'medical certificate' in reqs:
        return 'Medical Certificate'
    return np.NaN


def get_exam_type(in_str):
    """
    OPEN, INT_DEPT_PROM, DEPT_PROM, OPEN_INT_PROM
    
    Open only:
    >>> in_str = "ONLY ON AN OPEN COMPETITIVE BASIS"
    >>> get_exam_type(in_str)
    'OPEN'
    
    Interdepartmental only:
    >>> in_str = r"THIS EXAMINATION IS TO BE GIVEN ONLY\tON AN INTERDEPARTMENTAL PROMOTIONAL BASIS"
    >>> get_exam_type(in_str)
    'INT_DEPT_PROM'
    
    Departmental only:
    >>> in_str = "ONLY ON A DEPARTMENTAL PROMOTIONAL BASIS"
    >>> get_exam_type(in_str)
    'DEPT_PROM'
    
    Open and interdepartmental:
    >>> in_str = "INTERDEPARTMENTAL PROMOTIONAL AND AN OPEN COMPETITIVE BASIS"
    >>> get_exam_type(in_str)
    'OPEN_INT_PROM'
    """
    if pd.isna(in_str):
        return np.NaN
    
    m = re.search("ONLY\s*ON\s*AN\s*OPEN\s*COMPETITIVE\s*BASIS", in_str)
    if m:
        return 'OPEN'
    
    m = re.search("THIS\s*EXAMINATION\s*IS\s*TO\s*BE\s*GIVEN\s*ONLY\s*ON\s*AN\s*INTERDEPARTMENTAL\s*PROMOTIONAL\s*BASIS", in_str)
    if m:
        return 'INT_DEPT_PROM'
    
    m = re.search("ONLY\s*ON\s*A\s*DEPARTMENTAL\s*PROMOTIONAL\s*BASIS", in_str)
    if m:
        return 'DEPT_PROM'
    m = re.search("INTERDEPARTMENTAL\s*PROMOTIONAL\s*AND\s*AN\s*OPEN\s*COMPETITIVE\s*BASIS", in_str)
    if m:
        return 'OPEN_INT_PROM'
    return np.NaN


def extract_header(header):
    items = [item for item in header.split('\n')]
    items = [item.strip() for item in items]
    items = [item.split(':') for item in items if ':' in item]
    return {item[0]: item[1] for item in items}


def get_class_code(header):
    return extract_header(header).get('Class Code', np.NaN)


def get_open_date(header):
    return extract_header(header).get('Open Date', np.NaN)


# Find sections

def get_text_before_next_section(text):
    m = re.search('\s+([A-Z]{3,})+', text, re.DOTALL)
    start = m.start() if m else len(text)
    return text[:start]
#     return text[:re.search('\s+([A-Z]{3,})+', text, re.DOTALL).start()]


def find_sections(req):
    """
    >>> paragraph = '1. foo\\n2. bar'
    >>> [el.group() for el in find_sections(paragraph)]
    ['1. foo', '2. bar']
    """
    return re.finditer('[0-9]\..*', req)


# Find subsections
def find_subsections(req):
    """
    >>> paragraph = 'a. foo\\nb. bar'
    >>> [el.group() for el in find_subsections(paragraph)]
    ['a. foo', 'b. bar']
    """
    return re.finditer('[a-z]\..*', req)


def get_subsections_within_span(subsections, start, end):
    """
    """
    subsections_within_span = []

    for subsection in subsections:
        if subsection.start() > start and subsection.end() < end:
            subsections_within_span.append(subsection)
            
    return subsections_within_span


def get_requirement_set_id(in_str):
    """
    >>> in_str = '1. Graduation from an accredited four-year college or university with a major in Computer Science, Information Systems, or Geographical Information Systems; or'
    >>> get_requirement_set_id(in_str)
    '1'
    """
    m = re.search('(^[0-9](?=\.))', in_str)
    return m.group() if m else np.NaN


def get_requirement_set_description(in_str):
    """
    >>> in_str = '1. Graduation from an accredited four-year college or university with a major in Computer Science, Information Systems, or Geographical Information Systems; or'
    >>> get_requirement_set_description(in_str)
    'Graduation from an accredited four-year college or university with a major in Computer Science, Information Systems, or Geographical Information Systems; or'
    """
    m = re.search('(?<=^[0-9]\. ).*', in_str)
    return m.group() if m else np.NaN


def get_requirement_subset_id(in_str):
    """
    >>> in_str = 'a. the development, analysis, implementation or major modification of new or existing computer-based information systems or relational databases; or'
    >>> get_requirement_subset_id(in_str)
    'a'
    
    # >>> in_str = '(a)	Successful completion of the Animal Keeper Training program conducted by the Los Angeles Zoo; or '
    # >>> get_requirement_subset_id(in_str)
    """
    m = re.search('(^[a-z](?=\.))', in_str)
    return m.group() if m else np.NaN


def get_requirement_subset_description(in_str):
    """
    >>> in_str = 'a. the development, analysis, implementation or major modification of new or existing computer-based information systems or relational databases; or'
    >>> get_requirement_subset_description(in_str)
    'the development, analysis, implementation or major modification of new or existing computer-based information systems or relational databases; or'
    """
    m = re.search('(?<=^[a-z]\. ).*', in_str)
    return m.group() if m else np.NaN


def subsection_struct(subsections, start, end):
    bary = {}
    subsections_within = get_subsections_within_span(subsections, start, end)
    for el in subsections_within:
        bary[get_requirement_subset_id(el.group())] = get_requirement_subset_description(el.group())
    return bary


def subsections_to_df(req, header):
    req = get_text_before_next_section(req)
    sections = list(find_sections(req))
    if len(sections) == 0:
        return np.NaN
    subsections = list(find_subsections(req))

    fooy = defaultdict(dict)

    for head, tail in sliding_window(2, sections):
        set_id = get_requirement_set_id(head.group())
        fooy[set_id]['desc'] = get_requirement_set_description(head.group())
        fooy[set_id]['subsections'] = subsection_struct(subsections, head.end(), tail.start())


    last_section = sections[-1]
    set_id = get_requirement_set_id(last_section.group())
    fooy[set_id]['desc'] = get_requirement_set_description(last_section.group())
    fooy[set_id]['subsections'] = subsection_struct(subsections, last_section.end(), len(req) + 1)
    
    l = []

    for set_id in fooy.keys():
        if len(fooy[set_id]['subsections'].items()) == 0:
            l.append([header, set_id, np.NaN, np.NaN, fooy[set_id]['desc']])
        else:
            for el, foo in fooy[set_id]['subsections'].items():
                l.append([header, set_id, el, foo, fooy[set_id]['desc']])

    df = pd.DataFrame(l, columns=['JOB_CLASS_TITLE', 'REQUIREMENT_SET_ID', 'REQUIREMENT_SUBSET_ID', 'MISC_COURSE_DETAILS', 'REQ_OVERALL_DETAILS'])
    return df


def get_job_class_title(header):
    return re.split('\n|\t', header)[0]


def get_raw_fnames(fpaths):
    fnames = [x.name for x in fpaths]
    return fnames


def files_have_equal_content(fpath_orig, fpath_changed):
    hex_orig = get_hexdigest_of_fcontent(fpath_orig)
    hex_changed = get_hexdigest_of_fcontent(fpath_changed)
    return hex_orig == hex_changed


def output_is_equal_to_orig():
    print(files_have_equal_content('df_orig.csv', 'df_updated.csv'))
    print(files_have_equal_content('df_merged_orig.csv', 'df_merged_updated.csv'))
    

def get_education_major(in_str, list_of_majors):
    m = re.search('major in .*', in_str)
    if not m:
        return np.NaN
    in_str = in_str[m.start():m.end()]
    in_str.lower()
    majors = [major for major in list_of_majors if major in in_str]
    reiter = re.finditer('([A-Z][a-z]+)( [A-Z][a-z]+)*', in_str)
    reiter = [el.group() for el in reiter if el.group() not in ['Education Section', 'Education Department', 'Candidates', 'Personnel Department', 'City', 'City Application', 'Applicants']]
    majors.extend(reiter)
    majors = '|'.join([el.capitalize() for el in majors])
    return majors


def get_list_of_majors():
    list_of_majors = csv.reader(open('../input/list-of-majors/list_of_majors.csv'))
    list_of_majors = [line for line in list_of_majors if len(line) > 0]
    list_of_majors = [line for line in list_of_majors if not line[0].startswith('#')]
    list_of_majors = [el.strip() for el in flatten(list_of_majors)]
    list_of_majors = [el.strip('*') for el in list_of_majors]
    list_of_majors = [el.strip('&') for el in list_of_majors]
    list_of_majors = list(flatten([split_majors(el) for el in list_of_majors]))
    list_of_majors = [el.lower() for el in list_of_majors]
    return sorted(list_of_majors)


def split_majors(in_str):
    
    """
    No option:
    >>> in_str = 'Accounting Technician'
    >>> split_majors(in_str)
    ['Accounting Technician']
    
    Two options:
    >>> in_str = 'Labor/Industrial Relations'
    >>> split_majors(in_str)
    ['Labor Relations', 'Industrial Relations']
    
    Three options:
    >>> in_str = 'Parks/Rec/Leisure Facilities Management'
    >>> split_majors(in_str)
    ['Parks Facilities Management', 'Rec Facilities Management', 'Leisure Facilities Management']
    
    Other patterns could also be handled somewhat easily, but were not because of time constraints. Examples:
    'Adult Development & Aging/Gerontology', 'Wildlife & Wildlands Management',
    'Architectural Drafting/CAD Technology', 'Diesel Mechanics/Technology',
    'Exercise Science/Physiology/Kinesiology', 'Physical Therapy (Pre-Physical Therapy)',
    
    Some more cleanup would have been good too:
    'Communication Disorder Services (e.g.', 'Speech Pathology)'
    """
    options = re.match('\w+/\w+(/\w+)*', in_str)
    if not options:
        return [in_str]
    options = options.group(0)
    in_str = in_str.replace(options, '')
    in_str = in_str.strip()
    options = options.split('/')
    options = [f'{option} {in_str}' for option in options]
    return options
    

# === START: SECTION DETECTION


def load_job_bulletin(fpath):
    try:
        with open(fpath) as f:
            return f.read()
    except:
        with open(fpath, encoding='iso-8859-1') as f:
            return f.read()
        

def get_fnames(fpaths):
    fnames = (x.name for x in fpaths)
    fnames = (x.upper() for x in fnames)
    fnames = (re.search('[A-Z]*( [A-Z]*)+', x)[0] for x in fnames)
    fnames = (x.strip() for x in fnames)
    return fnames


def extract_content(s, fcontent):

    d = {'HEADER': fcontent[:s[0].start()]}

    for i, pair in enumerate(sliding_window(2, s[s.notnull()])):
        head, tail = pair
        d[s.index[i]] = fcontent[head.end():tail.start()]

    d[s.index[-1]] = fcontent[tail.end():-1]
    return d


def extract_section_by_regex(fcontent, fname):
    matches = {}
    matches['FNAME'] = [fname]
    for name, regex in REGEXES.items():
        matches[name] = regex.search(fcontent)
    return matches


def get_sections():
    fpaths = sorted(FOLDER_JOB_BULLETINS.iterdir())
    fcontents = [load_job_bulletin(fpath) for fpath in fpaths]
    fnames = list(get_fnames(fpaths))

    dfs = [extract_section_by_regex(fcontent, fname) for fcontent, fname in zip(fcontents, fnames)]
    dfs = [pd.DataFrame(df) for df in dfs]
    df_all = pd.concat(dfs)

    rows = [s[1] for s in df_all.iterrows()]
    rows = [s[s.notnull()] for s in rows]
    rows = [s[1:] for s in rows]
    rows = [extract_content(s, f) for s, f in zip(rows, fcontents)]
    return pd.DataFrame(rows)


def clean_columns(df):
    ALLOWED_COLUMNS = set(['FILE_NAME', 'JOB_CLASS_TITLE', 'JOB_CLASS_NO', 'REQUIREMENT_SET_ID',
                           'REQUIREMENT_SUBSET_ID', 'JOB_DUTIES', 'EDUCATION_YEARS', 'SCHOOL_TYPE',
                           'EDUCATION_MAJOR', 'FULL_TIME_PART_TIME', 'MISC_COURSE_DETAILS',
                           'DRIVERS_LICENSE_REQ', 'DRIV_LIC_TYPE', 'ADDTL_LIC', 'EXAM_TYPE',
                           'ENTRY_SALARY_GEN', 'ENTRY_SALARY_DWP', 'OPEN_DATE'])
    df.drop(columns=set(df.columns).difference(ALLOWED_COLUMNS))
    return df


# === END: SECTION DETECTION


def main(df):

    df['DRIVERS_LICENSE_REQ'] = df['REQS'].apply(get_drivers_license_info)
    df['DRIV_LIC_TYPE'] = df['REQS'].apply(get_drivers_license_class)
    df['ENTRY_SALARY_DWP'] = df['ANNUAL SALARY'].apply(get_dwp_salary)
    df['ADDTL_LIC'] = df['REQS'].apply(get_additional_licenses)
    df['EXAM_TYPE'] = df['SELECTION PROCESS'].apply(get_exam_type)
    df['FULL_TIME_PART_TIME'] = df['REQS'].apply(get_working_hours)
    df['SCHOOL_TYPE'] = df['REQS'].apply(get_education)
    df['ENTRY_SALARY_GEN'] = df['ANNUAL SALARY'].apply(get_salaries)

    df['JOB_CLASS_NO'] = df['HEADER'].apply(get_class_code)
    df['OPEN_DATE'] = df['HEADER'].apply(get_open_date)
    df['FILE_NAME'] = get_raw_fnames(sorted(FOLDER_JOB_BULLETINS.iterdir()))
    df['JOB_CLASS_TITLE'] = df['HEADER'].apply(get_job_class_title)
    df['EDUCATION_MAJOR'] = df['REQS'].apply(get_education_major, args=(LIST_OF_MAJORS,))

    headers = df['JOB_CLASS_TITLE']
    reqs = df['REQS']
    l_reqs = [subsections_to_df(req, header) for header, req in zip(headers, reqs)]
    l_reqs = [req for req in l_reqs if not isinstance(req, float)]
    df_subsections = pd.concat(l_reqs)
    df_subsections = df_subsections.reset_index()
    df_subsections[df_subsections['JOB_CLASS_TITLE'] == 'AIR CONDITIONING MECHANIC']
    df_merged = pd.merge(df, df_subsections, on='JOB_CLASS_TITLE', how='outer')
    df_merged[df_merged['JOB_CLASS_TITLE'] == 'SYSTEMS ANALYST']
    df_merged['EDUCATION_YEARS'] = df_merged['REQ_OVERALL_DETAILS'].apply(get_education_years)
    
    return df_merged


df = get_sections()
df_merged = main(df)

    
    
df_clean = df_merged.drop(columns=['ANNUAL SALARY', 'APPLICATION DEADLINE', 'HEADER', 'REQS', 'SELECTION PROCESS', 'WHERE TO APPLY', 'index', 'REQ_OVERALL_DETAILS'])
df_clean.rename(columns={'DUTIES': 'JOB_DUTIES'}, inplace=True)

column_order = ['FILE_NAME', 'JOB_CLASS_TITLE', 'JOB_CLASS_NO', 'REQUIREMENT_SET_ID', 'REQUIREMENT_SUBSET_ID', 'JOB_DUTIES', 'EDUCATION_YEARS', 'SCHOOL_TYPE', 'EDUCATION_MAJOR', 'FULL_TIME_PART_TIME', 'MISC_COURSE_DETAILS', 'DRIVERS_LICENSE_REQ', 'DRIV_LIC_TYPE', 'ADDTL_LIC', 'EXAM_TYPE', 'ENTRY_SALARY_GEN', 'ENTRY_SALARY_DWP', 'OPEN_DATE']
df_clean = df_clean[column_order]
df_clean.to_csv(csv_path)
df_clean
# Code for analysis
def get_salary(salary):
    if 'flat-rated' in salary:
        return int(df_clean['ENTRY_SALARY_GEN'][19].split()[0].replace('$', '').replace(',', ''))
    min_sal = int(salary.split(' - ')[0].strip('$').replace(',', ''))
    max_sal = int(salary.split(' - ')[1].strip('$').replace(',', ''))
    return mean([min_sal, max_sal])


def draw_barchart_education_time_required(df):
    df['EDUCATION_YEARS'] = df['EDUCATION_YEARS'].fillna(0)
    s_education_years = df.drop_duplicates('FILE_NAME').groupby('EDUCATION_YEARS')['FILE_NAME'].count()
    s_education_years.plot.bar(title='Year spent in education')


def make_network_between_salary_and_school_type(df):
    labels_income = ['inc_very_low', 'inc_low', 'inc_medium', 'inc_high', 'inc_very_high']
    s_salaries = df['ENTRY_SALARY_GEN'].fillna("0 - 0")
    df['ENTRY_SALARY_GEN_MEAN'] = s_salaries.apply(get_salary)

    school_types = df.groupby('FILE_NAME')['SCHOOL_TYPE'].last()
    school_types = school_types.fillna("Not available")
    
    # Get salary bins
    binned_salaries = df.groupby('FILE_NAME')['ENTRY_SALARY_GEN_MEAN'].max()
    binned_salaries_as_labels = pd.cut(binned_salaries, bins=5, labels=labels_income)
    binned_salaries_as_intervals = pd.cut(binned_salaries, bins=5)
    
    G = nx.MultiGraph()

    G.add_edges_from(zip(binned_salaries_as_labels, school_types))

    node_colors = ['blue' if node in set(school_types) else 'green' for node in G.nodes()]
    node_sizes = list(dict(G.degree()).values())
    node_sizes = [k * 5 for k in node_sizes]

    pos = nx.spring_layout(G,k=2.5,iterations=20)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes)
    
def make_network_between_salary_and_education_years(df):
    labels_income = ['inc_very_low', 'inc_low', 'inc_medium', 'inc_high', 'inc_very_high']
    s_salaries = df['ENTRY_SALARY_GEN'].fillna("0 - 0")
    df['ENTRY_SALARY_GEN_MEAN'] = s_salaries.apply(get_salary)
    
    # Get salary bins
    binned_salaries = df.groupby('FILE_NAME')['ENTRY_SALARY_GEN_MEAN'].max()
    binned_salaries_as_labels = pd.cut(binned_salaries, bins=5, labels=labels_income)
    binned_salaries_as_intervals = pd.cut(binned_salaries, bins=5)

    mapping_salary_bins_to_intervals = dict(zip(binned_salaries_as_labels, binned_salaries_as_intervals))

    # Get education bins
    labels_education = ['edu_very_short', 'edu_short', 'edu_medium', 'edu_long', 'edu_very_long']
    binned_education = df.groupby('FILE_NAME')['EDUCATION_YEARS'].max()
    binned_education_as_labels = pd.cut(binned_education, bins=5, labels=labels_education)
    binned_education_as_intervals = pd.cut(binned_education, bins=5)

    mapping_education_bins_to_intervals = dict(zip(binned_education_as_labels, binned_education_as_intervals))

    G = nx.MultiGraph()

    G.add_edges_from(zip(binned_salaries_as_labels, binned_education_as_labels))
    
    node_colors = ['blue' if node in labels_education else 'green' for node in G.nodes()]
    node_sizes = list(dict(G.degree()).values())
    node_sizes = [k * 5 for k in node_sizes]
    
    pos = nx.spring_layout(G,k=2.5,iterations=20)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes)
    

READING_LEVEL_BY_GRADE = {pd.Interval(left=13, right=float("inf"), closed='left'): 'College and University',
                          pd.Interval(left=9, right=12, closed='left'): 'High School',
                          pd.Interval(left=-float("inf"), right=8, closed='left'): 'Other',}


def get_grade_of_reading_level(score):
    for iv in READING_LEVEL_BY_GRADE.keys():
        if score in iv:
            return READING_LEVEL_BY_GRADE[iv]
        

def get_required_reading_skills(df):
    df['SCHOOL_TYPE'] = df['SCHOOL_TYPE'].fillna('Not available')
    job_duties = df['JOB_DUTIES']
    job_duties = job_duties.fillna('empty')
    job_duties.replace("", "empty", inplace=True)
    job_duties = job_duties.apply(word_tokenize)
    job_duties = job_duties.apply(readability.getmeasures)
    job_duties = job_duties.apply(lambda x: x['readability grades']['GunningFogIndex'])
    job_duties = job_duties.apply(get_grade_of_reading_level)

    df['GUNNING_FOG_INDEX'] = job_duties.values

    reading_levels_required = df.groupby(['SCHOOL_TYPE', 'GUNNING_FOG_INDEX']).size()
    return reading_levels_required


def draw_barchart_for_required_reading_skills(df):
    reading_levels_required = get_required_reading_skills(df)
    reading_levels_required.unstack(level=0).T.plot(kind='bar')
    

def draw_heatmap_for_required_reading_skills(df):
    reading_levels_required = get_required_reading_skills(df)
    df_reading_levels = reading_levels_required.unstack(level=0).T.fillna(0)
    df_reading_levels = df_reading_levels.astype(int)
    df_reading_levels = df_reading_levels.T

    sns.set()
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df_reading_levels, annot=True, fmt="d", linewidths=.5, ax=ax)
draw_barchart_education_time_required(df_clean)
make_network_between_salary_and_education_years(df_clean)
make_network_between_salary_and_school_type(df_clean)
draw_barchart_for_required_reading_skills(df_clean)
draw_heatmap_for_required_reading_skills(df_clean)
