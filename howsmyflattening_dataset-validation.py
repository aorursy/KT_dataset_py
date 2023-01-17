import csv
import unittest
 
validate = [
    ['canada_mortality', 10],
    ['canada_recovered', 4],
    ['icu_capacity_on', 4],
    ['icucapacity', 14],
    ['international_mortality', 4],
    ['international_recovered', 4],
    ['npi_canada', 25],
    ['test_data_canada', 10],
    ['test_data_intl', 4],
    ['test_data_on', 11],
    ['vis_canada_mobility', 5],
    ['vis_growth', 3],
    ['vis_growthrecent', 5],
    ['vis_icucapacity', 20],
    ['vis_icucapacityprovince', 19],
    ['vis_icucasestatusprovince', 3],
    ['vis_phu', 3],
    ['vis_results', 3],
    ['vis_testresults', 17]
]

inputpath = '/kaggle/input/covid19-challenges/'

def isempty(csv):
    with open(csv) as f:
        for i, l in enumerate(f):
            if i == 1:
                return False
    return True

def columncount(csv):
    with open(csv) as f:
        line = f.readline()
        ncol = len(line.split(','))
        return ncol
    return 0

class FileTest(unittest.TestCase):
    def setUp(self):
        self.errors = []
    def tearDown(self):
        if len(self.errors) > 0:
            print('All Errors:\n{}'.format('\n'.join(self.errors)))
        self.assertEqual([], self.errors)
    def test_init(self):
        for test in validate:
            csv = test[0] + '.csv'
            filename = inputpath+csv
            try:self.assertEqual(isempty(filename), False)
            except: self.errors.append('{} is empty'.format(csv))
            try:self.assertEqual(columncount(filename), test[1])
            except: self.errors.append('{} has incorrect number of columns'.format(csv))
            
if __name__ == "__main__":    
    unittest.main(argv=['first-arg-is-ignored'], exit=False)