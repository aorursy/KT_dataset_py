# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from json import load, dump
from os.path import sep
true, false, null = True, False, None
keywords = ['False', 'None', 'True', 'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else',
            'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or',
            'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
code = ''
identifiers = []
line = 0
LineLengthCalculator = []
LineWordCalculator = []
IdentifierLengthCalculator = []
InlineSpaceTab = []
TrailingSpaceTab = []
UnderScoreCalculator = []
IndentSpaceTab = []
llc, lwc, inline_space_tab, trailing_space_tab, indent_space_tab = [0] * 5
CommentFrequencyCalculator = (0, 0)
def reset_metrics():
    global identifiers, LineLengthCalculator, LineWordCalculator, IdentifierLengthCalculator
    global IndentSpaceTab, TrailingSpaceTab, UnderScoreCalculatorer
    global IdentifierLengthCalculator, CommentFrequencyCalculator
    identifiers = []
    LineLengthCalculator = []
    LineWordCalculator = []
    IdentifierLengthCalculator = []
    InlineSpaceTab = []
    TrailingSpaceTab = []
    UnderScoreCalculator = []
    IndentSpaceTab = []
    CommentFrequencyCalculator = (0, 0)
def get_string_at(i):
    j = i + 1
    while j < len(code):
        ch = code[j]  # type: str
        if not (ch.isalnum() or ch == '_'):
            return code[i: j]
        j += 1
    return code[i: j]

def check_key_word(word):
    return word in keywords

def handle_identifier_at(i):
    word = get_string_at(i)
    i += (len(word) - 1)

    is_key_word = check_key_word(word)
    if not is_key_word:
        IdentifierLengthCalculator.append(len(word))

        under_count = word.count('_')
        UnderScoreCalculator.append(under_count)
    return i

def handle_number_at(i):
    j = i + 1
    got_dot = (code[i] == '.')

    while j < len(code):
        ch = code[j]
        if not ch.isdecimal():
            if ch == '.':
                if got_dot:
                    break
                got_dot = true
            else:
                break
        j += 1
    return j - 1


def handle_block_comment_at(i):
    start = code[i: i + 3]
    end = code.find(start, i + 3)
    return end + 2  # end of comment literal

def handle_string_at(i):
    global llc
    start = code[i: i + 3]
    if start == '"""' or start == "'''":
        return handle_block_comment_at(i), true
    start = code[i]
    j = i
    while j < len(code) - 1:
        j += 1
        llc += 1
        ch = code[j]
        if ch == '\n':
            LineLengthCalculator.append(llc)
            llc = 0
        if ch != start:
            continue
        if code[j - 1] != '\\':
            return j, false
        elif code[j - 2: j] == '\\\\':
            return j, false
    return j, false

def handle_equalable_at(i):
    if code[i + 1] == '=':
        return i + 1
    return i

def handle_line_comment_at(i):
    j = i + 1
    while j < len(code):
        if code[j] == '\n':
            return j - 1
        j += 1
    return j - 1
from time import time, gmtime, strftime
def get_metrics():
    global code, line
    global llc, lwc, inline_space_tab, trailing_space_tab, indent_space_tab, CommentFrequencyCalculator

    tick = time()
    line = 1
    beginning_of_line = true
    llc, lwc, inline_space_tab, trailing_space_tab, indent_space_tab = [0] * 5
    CommentFrequencyCalculator = [0, 0]
    i = 0
    while i < len(code):  # type: int
        ch = code[i]  # type: str

        # check time
        tock = time()
        if tock - tick > 30:
            raise RuntimeError("Execution took {:.2f} secods".format(tock - tick))
        
        llc += 1
        if ch.isspace():
            if beginning_of_line:
                indent_space_tab += 1
            else:
                trailing_space_tab += 1
            if ch == '\n':
                LineLengthCalculator.append(llc)
                llc = 0
                LineWordCalculator.append(lwc)
                lwc = 0
                InlineSpaceTab.append(inline_space_tab)
                inline_space_tab = 0
                TrailingSpaceTab.append(trailing_space_tab)
                trailing_space_tab = 0
                IndentSpaceTab.append(indent_space_tab)
                indent_space_tab = 0

                beginning_of_line = true
                line += 1

            i += 1
            continue

        beginning_of_line = false
        lwc += 1

        # If more alphanum found, trailing space-tabs are actually inline space-tab
        inline_space_tab += trailing_space_tab
        trailing_space_tab = 0
        if ch.isalpha():
            j = handle_identifier_at(i)
        elif ch.isdecimal() or ch == '.':
            j = handle_number_at(i)
        elif ch == '"' or ch == "'":
            j, is_blk_cmnt = handle_string_at(i)
            if is_blk_cmnt:
                CommentFrequencyCalculator[1] += 1
                llc -= (j - i)

        elif ch == '<' or ch == '>' or ch == '!' or ch == '=' or ch == '+' or ch == '-' or ch == '*' or ch == '/':
            j = handle_equalable_at(i)

        # Line Comment
        elif ch == '#':
            CommentFrequencyCalculator[0] += 1
            j = handle_line_comment_at(i)
        # Nothing else matters
        else:
            j = i

        llc += (j - i)
        i = j

        i += 1

base_dir = "../input/"
with open("{}py150_files/python100k_train.txt".format(base_dir)) as file:
    data_summary = file.read().split("\n")
del data_summary[-1]
total_files = len(data_summary)
total_files
data_summary[-1]
def author_name(file_name):
    first = file_name.index('/')
    second = file_name.index('/', first+1)
    return file_name[first+1: second]
skip_list = [
    'data/bchartoff/regexcalibur/screens/__init__.py',  # IndexError
    'data/cpfair/tapiriik/tapiriik/services/TrainingPeaks/__init__.py',
    'data/dailymuse/oz/oz/skeleton/plugin/tests/__init__.py',
    'data/jamstooks/django-s3-folder-storage/s3_folder_storage/tests/__init__.py',
    'data/machinalis/quepy/quepy/quepyapp.py',  # Execution Time
    'data/ml-lab/auxiliary-deep-generative-models/models/__init__.py',  # IndexError
    'data/nhfruchter/pgh-bustime/pghbustime/__init__.py',
    'data/python-excel/xlrd/xlrd/biffh.py',  #UnicodeDecodeError
    'data/rotoudjimaye/django-jy/src/main/python/djangojy/db/backends/util.py',  #IndexError
    'data/babble/babble/include/jython/Lib/getopt.py',  # UnicodeDecodeError
    'data/cloudera/hue/desktop/core/ext-py/tablib-0.10.0/tablib/packages/xlrd/formatting.py',
    'data/cpfair/tapiriik/tapiriik/services/Motivato/__init__.py',  # Index
    'data/dragondjf/PFramer/qframer/qt/QtWidgets.py',
    'data/dziegler/django-cachebot/cachebot/models.py',
    'data/francelabs/datafari/windows/python/Tools/i18n/msgfmt.py',  # Unicode
    'data/Komodo/KomodoEdit/contrib/html5lib/utils/entities.py',  # Execution Time
    'data/pybrain/pybrain/pybrain/rl/explorers/__init__.py',  #Index
    'data/pydanny/django-wall/wall/tests/__init__.py',
    'data/Gallopsled/pwntools-write-ups/wargames/overthewire-vortex/level4/libformatstr/__init__.py',
    'data/OSCAAR/OSCAAR/oscaar/astrometry/__init__.py',
    'data/adafruit/Adafruit_Nokia_LCD/Adafruit_Nokia_LCD/__init__.py',
    'data/babble/babble/include/jython/Lib/test/test_csv.py',  # Unicode
    'data/bradleyfay/py-Goldsberry/goldsberry/league/__init__.py',  # Index
    'data/django-leonardo/django-leonardo/tests/testapp/tests/__init__.py',
    'data/dropbox/pyston/from_cpython/Lib/test/test_mailbox.py',  # Execution Time
    'data/ctxis/canape/CANAPE.Scripting/Lib/sqlite3/dbapi2.py',  # Unicode
    'data/fp7-ofelia/ocf/ofreg/registration/models/__init__.py',  # Index
    'data/glue-viz/glue/glue/dialogs/link_editor/qt/__init__.py',  #Index
    'data/glue-viz/glue/glue/qt/widget_properties.py', 
    'data/mclarkk/lifxlan/lifxlan/__init__.py',
    'data/mozilla/inventory/vendor-local/src/django-tastypie/tests/basic/tests/__init__.py',
    'data/nextml/NEXT/next/apps/CardinalBanditsPureExploration/algs/RoundRobin/__init__.py',
    'data/acba/elm/elm/__init__.py',
    'data/amrdraz/kodr/app/brython/www/src/Lib/browser/html.py',
    'data/bogdan-kulynych/trials/trials/__init__.py',
    'data/pokitdok/pokitdok-python/tests/__init__.py',
    'data/samdmarshall/xcparse/xcparse/Xcode/BuildSystem/Environment/__init__.py',
    'data/shuge/Qt-Python-Binding-Examples/qcommons/__init__.py',
    'data/DataDog/brod/brod/__init__.py',
    'data/bradleyfay/py-Goldsberry/goldsberry/draft/__init__.py',
    'data/clione/django-kanban/src/kanban/settings/__init__.py',
    'data/dcramer/django-indexer/indexer/tests/__init__.py',
    'data/django-oscar/django-oscar/tests/_site/apps/customer/models.py',
    'data/django/djangobench/djangobench/benchmarks/form_clean/settings.py',
    'data/rvanlaar/easy-transifex/src/transifex/transifex/txcommon/tests/__init__.py',
    'data/wesabe/fixofx/3rdparty/wsgi_intercept/urllib2_intercept/__init__.py',
    'data/yanapermana/metadecryptor/lib3.py',
    'data/zmap/ztag/ztag/protocols.py',
    'data/cpfair/tapiriik/tapiriik/services/SportTracks/__init__.py',
    'data/dailymuse/oz/oz/redis_sessions/tests/__init__.py',
    'data/dndtools/dndtools/dndtools/dndproject/settings.py',
    'data/glue-viz/glue/glue/viewers/histogram/qt/__init__.py',
    'data/machinalis/quepy/tests/testapp/__init__.py',
    'data/ml-brasil/discover-github-data/octograb/preselection/__init__.py',
    'data/VisualComputingInstitute/Beacon8/beacon8/criteria/__init__.py',
    'data/flosch/simpleapi/simpleapi/message/__init__.py',
    'data/glue-viz/glue/glue/dialogs/custom_component/qt/__init__.py',
    'data/glue-viz/glue/glue/viewers/custom/qt/__init__.py',
    'data/nextml/NEXT/next/database_client/PermStore/__init__.py',
    'data/pybrain/pybrain/pybrain/optimization/__init__.py',
    'data/debrouwere/django-locking/locking/tests/__init__.py',
    'data/KanColleTool/kcsrv/db/__init__.py',
    'data/MongoEngine/mongoengine/tests/queryset/__init__.py',
    'data/amrdraz/kodr/app/brython/www/src/Lib/browser/svg.py',
    'data/ericholscher/devmason-server/devmason_server/tests/__init__.py',
    'data/networkdynamics/zenlib/src/zen/algorithms/flow/__init__.py',
    'data/pybrain/pybrain/pybrain/rl/environments/ode/__init__.py',
    'data/sahana/eden/modules/tests/volunteer/__init__.py',
    'data/statsmodels/statsmodels/statsmodels/datasets/fertility/__init__.py',
    'data/tethysplatform/tethys/tethys_sdk/gizmos.py',
    'data/braceio/localdev/localdev/__init__.py',
    'data/nextml/NEXT/next/apps/CardinalBanditsPureExploration/algs/LUCB/__init__.py',
    'data/nextml/NEXT/next/apps/DuelingBanditsPureExploration/algs/BeatTheMean/__init__.py',
    'data/smart-classic/smart_server/smart/views/__init__.py',
    'data/springmeyer/djmapnik/djmapnik/__init__.py',
    'data/Akagi201/learning-python/flask/flask-boost/test/application/forms/__init__.py',
    'data/luzexi/xls2lua/xls2lua.py',  # Execution Time
    'data/chalasr/Flask-P2P/venv/lib/python2.7/site-packages/gunicorn/http/message.py',
    'data/toumorokoshi/sprinter/sprinter/core/tests/test_manifest.py',
    'data/mitsuhiko/python-modernize/libmodernize/fixes/fix_unicode.py',
    'data/RoseOu/flasky/venv/lib/python2.7/site-packages/gunicorn/http/message.py',
    'data/michaelliao/transwarp/transwarp/web.py',
    'data/andresgsaravia/research-engine/lib/pygments/lexers/felix.py',
    'data/kivy/python-for-android/pythonforandroid/bootstraps/pygame/build/build.py',
    'data/ARM-software/workload-automation/wlauto/utils/misc.py',
    'data/cloudera/hue/desktop/core/ext-py/pysqlite/pysqlite2/test/__init__.py',  # Unicode
    'data/cloudera/hue/desktop/core/ext-py/tablib-0.10.0/tablib/packages/xlrd/biffh.py',
    'data/deanhiller/databus/webapp/play1.3.x/python/Lib/msilib/__init__.py',
    'data/jpm/papercut/storage/forwarding_proxy.py',
    'data/Flolagale/mailin/python/DNS/Lib.py',
    'data/cloudera/hue/desktop/core/ext-py/pysqlite/pysqlite2/test/dbapi.py',
    'data/dropbox/pyston/from_cpython/Lib/sqlite3/dbapi2.py',
    'data/dropbox/pyston/from_cpython/Lib/sqlite3/test/transactions.py',
    'data/hydroshare/hydroshare2/hs_docker_base/pysqlite-2.6.3/build/lib.macosx-10.9-intel-2.7/pysqlite2/dbapi2.py',
    'data/cloudera/hue/desktop/core/ext-py/tablib-0.10.0/tablib/packages/xlrd/formula.py',
    'data/kennethreitz/tablib/tablib/packages/xlrd/examples/xlrdnameAPIdemo.py',
    'data/dropbox/pyston/from_cpython/Lib/plat-mac/macerrors.py',
    'data/hydroshare/hydroshare2/hs_docker_base/pysqlite-2.6.3/build/lib.macosx-10.9-intel-2.7/pysqlite2/test/transactions.py',
    
    'data/dropbox/pyston/from_cpython/Lib/distutils/util.py',  # Execution Time
    'data/dropbox/pyston/from_cpython/Lib/msilib/__init__.py',  # Unicode
    'data/dropbox/pyston/from_cpython/Lib/sqlite3/test/dbapi.py',
    'data/cloudera/hue/desktop/core/ext-py/pysqlite/pysqlite2/test/userfunctions.py',
    'data/twisted/twisted/twisted/test/test_pcp.py' # Runtime
]
'''
{'AppScale': 1898,
 'Azure': 438,
 'BU-NU-CLOUD-SP16': 501,
 'CenterForOpenScience': 433,
 'CollabQ': 397,
 'GoogleCloudPlatform': 712,
 'StackStorm': 417,
 'anandology': 435,
 'anhstudios': 5069,
 'azoft-dev-team': 329,
 'cloudera': 1028,
 'dimagi': 785,
 'django': 605,
 'dropbox': 801,
 'enthought': 700,
 'kuri65536': 391,
 'fp7-ofelia': 342,
 'freenas': 325,
 'getsentry': 439,
 'google': 621,
 'mozilla': 752,
 'openstack': 3111,
 'saltstack': 593,
 'sympy': 315,
 'tav': 852}
'''

top_thirty_five = [
    'AppScale', 'Azure', 'BU-NU-CLOUD-SP16', 'CenterForOpenScience', 
    'CiscoSystems', 'CollabQ', 'GoogleCloudPlatform', 'ImageEngine', 'OpenMDAO', 
    'StackStorm', 'VisTrails', 'anandology', 'anhstudios', 'azoft-dev-team', 'boto', 
    'cloudera', 'dimagi', 'django', 'dropbox', 'enthought', 'kuri65536', 'fp7-ofelia', 
    'freenas', 'getsentry', 'google', 'googleads', 'jmcnamara', 'mozilla', 'openstack', 
    'pantsbuild', 'robotframework', 'saltstack', 'sympy', 'tav', 'twisted'
]

top_twenty_five = [
    'AppScale', 'Azure', 'BU-NU-CLOUD-SP16', 'CenterForOpenScience', 'CollabQ',
    'GoogleCloudPlatform', 'StackStorm', 'anandology', 'anhstudios', 'azoft-dev-team',
    'cloudera', 'dimagi', 'django', 'dropbox', 'enthought', 'kuri65536', 'fp7-ofelia',
    'freenas', 'getsentry', 'google', 'mozilla', 'openstack', 'saltstack', 'sympy', 'tav'
]

top_ten = [
    'AppScale', 'GoogleCloudPlatform', 'anhstudios', 'cloudera', 'dimagi',
    'dropbox', 'enthought', 'mozilla', 'openstack', 'tav',
]

top_two = ['AppScale', 'cloudera']

selected_authors = {
    'freenas', 'Azure', 'GoogleCloudPlatform', 'StackStorm', 
    'dimagi', 'fp7-ofelia', 'enthought', 'sympy'
}
output = {}
i = 0
tick = time()
selected_total_train_files = 4034

for index, code_file_name in enumerate(data_summary):
    if len(code_file_name) != len(code_file_name.encode()):  # if not ascii filename
        continue
    
    author = author_name(code_file_name)
    if author not in selected_authors:
        continue
    if author not in output:
        output[author] = []
    
    reset_metrics()
    i += 1

    if i % 100 == 0:
        print("\r{:.2f} percent complete.".format((i / selected_total_train_files) * 100), end='')

    code_file_name = code_file_name.replace("/", sep)
    if code_file_name in skip_list:
        continue
    
    with open("{}data/{}".format(base_dir, code_file_name)) as code_file:
        try:
            code = code_file.read()
        except UnicodeDecodeError as err:
            raise RuntimeError("[{}] '{}': {}".format(index, code_file_name, err))

    try:
        get_metrics()
    except BaseException as err:
        raise RuntimeError("[{}] '{}': {}".format(index, code_file_name, err))

    metrics = {
        "line_lengths": LineLengthCalculator,
        "line_word_lengths": LineWordCalculator,
        "comment_counts": CommentFrequencyCalculator,
        "identifier_lengths": IdentifierLengthCalculator,
        "underscore_counts": UnderScoreCalculator,
        "indent_space_tab_lengths": IndentSpaceTab,
        "inline_space_tab_lengths": InlineSpaceTab,
        "trailing_space_tab_lengths": TrailingSpaceTab,
    }
    output[author].append(metrics)

tock = time()
print("\r{} percent complete.              ".format(100))
print("Completed in", strftime('%M: %S', gmtime(tock - tick)))

print("Number of authors:", len(output))
# 650 for 10 author
# 310 for 25 author
filterd_output = {k: output[k] for k in output}

print(len(filterd_output), "Authors")
{k: len(filterd_output[k]) for k in filterd_output}
total_filtered_files = sum([len(filterd_output[k]) for k in filterd_output])
total_filtered_files
LL_SIZE = 120
LW_SIZE = 20
IDL_SIZE = 20
UC_SIZE = 2
IDST_SIZE = 20
IST_SIZE = 10
TST_SIZE = 2
        
def prep(unq, cnt, size):
    feature = [0] * size
    
    i = 0
    for j, val in enumerate(unq):
        if i == size - 1:
            feature[i] = sum(cnt[j:])
            return feature
        
        while i < val:
            feature[i] = 0
            i += 1
            if i == size - 1:
                feature[i] = sum(cnt[j:])
                return feature
            
        feature[i] = cnt[j]
        i += 1
        
    return feature
features = {}
i = 0

tick = time()
for author in filterd_output:
    features[author] = []
    for metrics in filterd_output[author]:
        i += 1
        tock = time()

#         if i % 100 == 0:
        log = "{:.2f}% ({}/{})".format(
            (i / total_filtered_files) * 100, i, total_filtered_files
        )
        elapsed = strftime('%H: %M: %S', gmtime(tock - tick))
        log = f"\r{log} | Elapes: {elapsed}"
        print(log, end='')
            
        feature = {}
        
        ll_unq, ll_cnt = np.unique(metrics["line_lengths"], return_counts=True)
        lw_unq, lw_cnt = np.unique(metrics["line_word_lengths"], return_counts=True)
        idl_unq, idl_cnt = np.unique(metrics["identifier_lengths"], return_counts=True)
        uc_unq, uc_cnt = np.unique(metrics["underscore_counts"], return_counts=True)
        idst_unq, idst_cnt = np.unique(metrics["indent_space_tab_lengths"], return_counts=True)
        ist_unq, ist_cnt = np.unique(metrics["inline_space_tab_lengths"], return_counts=True)
        tst_unq, tst_cnt = np.unique(metrics["trailing_space_tab_lengths"], return_counts=True)
            
        feature["line_lengths"] = prep(ll_unq, ll_cnt, LL_SIZE)
        feature["line_word_lengths"] = prep(lw_unq, lw_cnt, LW_SIZE)
        feature["identifier_lengths"] = prep(idl_unq, idl_cnt, IDL_SIZE)
        feature["underscore_counts"] = prep(uc_unq, uc_cnt, UC_SIZE)
        feature["indent_space_tab_lengths"] = prep(idst_unq, idst_cnt, IDST_SIZE)
        feature["inline_space_tab_lengths"] = prep(ist_unq, ist_cnt, IST_SIZE)
        feature["trailing_space_tab_lengths"] = prep(tst_unq, tst_cnt, TST_SIZE) 
        feature["comment"] = list(metrics["comment_counts"])
        features[author].append(feature)

tock = time()
print("\r100 percent complete.                                  ")
print("Completed in", strftime('%H: %M: %S', gmtime(tock - tick)))
columns = ["ll" + str(i) for i in range(LL_SIZE)]
columns += ["lw" + str(i) for i in range(LW_SIZE)]
columns += ["ldl" + str(i) for i in range(IDL_SIZE)]
columns += ["uc" + str(i) for i in range(UC_SIZE)]
columns += ["idst" + str(i) for i in range(IDST_SIZE)]
columns += ["ist" + str(i) for i in range(IST_SIZE)]
columns += ["tst" + str(i) for i in range(TST_SIZE)]
columns += ["line_comment", "block_comment"]
columns += ["Author"]

data = []
for author in features:
    for feature in features[author]:
        row = (feature["line_lengths"] + feature["line_word_lengths"] + 
            feature["identifier_lengths"] + feature["underscore_counts"] + 
            feature["indent_space_tab_lengths"] + feature["inline_space_tab_lengths"] + 
            feature["trailing_space_tab_lengths"] + feature["comment"] + [author])
        data.append(row)
        
print("Completed.")
data_frame = pd.DataFrame(data, columns=columns)
data_frame.head()
data_frame.shape
data_frame.to_csv("selected_authors_train.csv")