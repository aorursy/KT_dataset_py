import marbles.core
from marbles.mixins import mixins

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TimeSeriesTestCase(marbles.core.TestCase,mixins.MonotonicMixins):
    def setUp(self):
        self.df = pd.DataFrame({'dates':[datetime(2018,1,1),
                                         datetime(2018,2,1),
                                         datetime(2018,2,1)],
                                'ireland_unemployment':[6.2,6.1,6.0]})
        
    def tearDown(self):
        self.df = None
        
    def test_date_order(self):
        
        self.assertMonotonicIncreasing(sequence=self.df.dates,
                                  note = 'Dates need to increase monotonically')
        
if __name__ == '__main__':
    marbles.core.main(argv=['first-arg-is-ignored'], exit=False)
