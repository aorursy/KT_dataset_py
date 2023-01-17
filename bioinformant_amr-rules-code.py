import yaml

import os



def read_yaml_files(directory):

  files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]



  data_collection = {}

  for file in files:

    file_suffix, ext = os.path.splitext(file)

    with open(os.path.join(directory, file)) as fh:

        data = yaml.load(fh, Loader=yaml.FullLoader)

        data_collection[file_suffix] = data

  return data_collection



# read in rules

rules_directory = '../input/amr-rules'

rules = read_yaml_files(rules_directory)

print(rules)



# read in determinant groups

determinant_groups_directory = '../input/amr-determinant-groups'

determinant_groups = read_yaml_files(determinant_groups_directory)

print(determinant_groups)



suppressor_groups_directory = '../input/amr-suppressor-groups'

suppressor_groups = read_yaml_files(suppressor_groups_directory)

print(suppressor_groups)
import re



def test_rules(rules, isolate_determinants):

  matching_rules = []

  any_match = False

  # check each rule

  for rule in rules:

    # for a rule each determinant group needs to have a match

    match_all_determinant_groups = True

    # check each determinant group

    for determinant_group in rule['determinant_groups']:

      determinant_patterns = determinant_groups[determinant_group['name']]

      # check all possible determinants for a group e.g NDM-[0-9]+, KPC-[0-9+]

      # if any match it's a match for the group

      determinant_group_match = False

      for determinant_pattern_string in determinant_patterns:

        determinant_pattern = re.compile(determinant_pattern_string)

        if any(determinant_pattern.match(determinant) for determinant in isolate_determinants):

          determinant_group_match = True

      # Now check suppressors

      suppressor_match = False

      if determinant_group['suppressors']:

        for suppressor_group in determinant_group['suppressors']:

          suppressor_patterns = suppressor_groups[suppressor_group]

          suppressor_group_match = False

          for suppressor_pattern_string in suppressor_patterns:

            suppressor_pattern = re.compile(suppressor_pattern_string)

            if any(suppressor_pattern.match(determinant) for determinant in isolate_determinants):

              suppressor_group_match = True

          if suppressor_group_match:

            suppressor_match = True

            

      if not determinant_group_match or suppressor_match:

        match_all_determinant_groups = False

    

    if match_all_determinant_groups:  

      matching_rules.append(",".join([group['name'] for group in rule['determinant_groups']]))

      any_match = True

  return(any_match, ";".join(matching_rules))    



isolate_1_determinants = ['NDM-1']

isolate_2_determinants = ['CTX-M-15', 'ompk36-mut1']

isolate_3_determinants = ['SHV-12']

isolate_4_determinants = ['SHV-12', 'ompk36-loss']

isolate_5_determinants = ['SHV-12', 'ompk36-loss', 'ompk36-suppressor-mut1']

carbapenem_rules = rules['carbapenems']





match, matching_rules = test_rules(carbapenem_rules, isolate_1_determinants)

print(match, matching_rules)



match, matching_rules = test_rules(carbapenem_rules, isolate_2_determinants)

print(match, matching_rules)



match, matching_rules = test_rules(carbapenem_rules, isolate_3_determinants)

print(match, matching_rules)



match, matching_rules = test_rules(carbapenem_rules, isolate_4_determinants)

print(match, matching_rules)



match, matching_rules = test_rules(carbapenem_rules, isolate_5_determinants)

print(match, matching_rules)