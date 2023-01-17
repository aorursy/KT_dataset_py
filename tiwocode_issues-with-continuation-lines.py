import numpy as np

import pandas as pd

import collections
emails = pd.read_csv("../input/emails.csv")
class IgnorantMessageParser:

    

    checks = ['head terminated by blank line']

    

    def __init__(self, data, *ID):

        self.ID=ID

        self.errors = collections.defaultdict(list)

        

        self._process_data(data)

        

    def _log_error(self, error, *description):

        self.errors[error].append(description)

        

    def _process_data(self, data):

        try:

            head, body = data.split('\n\n', 1)

        except ValueError:

            self._log_error('unterminated head')

            head, body = data, ''

        

        self.process_head(head)

        self.process_body(body)

        

    def process_head(self, head):

        ...



    def process_body(self, body):

        ...
class Checker:

    def __init__(self, parserclass):

        self.parse = parserclass

        self.errorneous_messages = []

        self.issues = collections.Counter()

        self.examples = {}

        self.n = 0

        

    def __call__(self, dataframe):

        for n, (index, file_, message) in enumerate(emails.itertuples()):

            parsed = self.parse(message, index)

            if parsed.errors:

                self.errorneous_messages.append(parsed)

                #self.errorneous_messages.append(index)

                for error in parsed.errors:

                    self.issues[error] += 1

                    self.examples[error] = parsed

            yield parsed

        self.n += n

    

    def _repr_html_(self):

        checks = '<span>(checks: ' + ', '.join(self.parse.checks) + ')</span>'

        if self.n:

            if self.errorneous_messages:

                issues = '<span>Issues found: ' + ', '.join(sorted(self.issues)) + '</span>'

                return ('<span class="Checker Checker_warning">⚠️ {}/{} messages with issues. {}.'

                        ' {}</span>'

                        ).format(len(self.errorneous_messages), self.n, checks, issues)

            else:

                return ('<span class="Checker Checker_ok">✔️ no issues found, {} messages checked. {}'

                        .format(self.n, checks))

        else:

            return ('<span class="Checker Checker_fresh">❗ no messages checked yet. {}</span>'

                        .format(checks))

            

            
def consume(iterator):

    collections.deque(iterator, maxlen=0)



checker = Checker(IgnorantMessageParser)

consume(checker(emails))

checker
class HeaderParser(IgnorantMessageParser):

    

    checks = ['head terminated by blank line', 'continuation lines']

    

    def __init__(self, data, *ID):

        self.data = data

        self.headers = collections.defaultdict(list)

        super().__init__(data, *ID)

    

    def process_head(self, head):

    

        currentname, currentvalue = None, None

        for line in head.splitlines():

            if line.startswith('\t') or line.startswith(' '):

                if currentname is None:

                    self._log_error('continuation in first line', line)

                else:

                    currentvalue.append(line.strip())

            elif ':' in line:

                if currentname is not None:

                    self.process_header(currentname.lower(), ' '.join(currentvalue))

                currentname, currentvalue = line.split(':', 1)

                currentvalue = [currentvalue]

            elif line == '':

                break

            else:

                self._log_error('unindented continuation', line)

                if currentname is None:

                    self._log_error('continuation in first line', line)

                else:

                    currentvalue.append(line.strip())     

    

    def process_header(self, name, value):

        self.headers[name].append(value)
checker = Checker(HeaderParser)

headers_seen = set()

for msg in checker(emails):

    headers_seen.update(msg.headers)

checker
sorted(headers_seen)