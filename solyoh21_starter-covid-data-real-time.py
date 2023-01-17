"""UserSecret client classes.
This library adds support for communicating with the UserSecrets service,
currently used for retrieving an access token for supported integrations
(ie. BigQuery).
"""
 
import json
import os
import socket
import urllib.request
from datetime import datetime, timedelta
from enum import Enum, unique
from typing import Optional, Tuple
from urllib.error import HTTPError, URLError
 
_KAGGLE_DEFAULT_URL_BASE = "https://www.kaggle.com"
_KAGGLE_URL_BASE_ENV_VAR_NAME = "KAGGLE_URL_BASE"
_KAGGLE_USER_SECRETS_TOKEN_ENV_VAR_NAME = "KAGGLE_USER_SECRETS_TOKEN"
TIMEOUT_SECS = 40
 
 
class CredentialError(Exception):
    pass
 
 
class BackendError(Exception):
    pass
 
 
class ValidationError(Exception):
    pass
 
@unique
class GcpTarget(Enum):
    """Enum class to store GCP targets."""
    BIGQUERY = (1, "BigQuery")
    GCS = (2, "Google Cloud Storage")
    AUTOML = (3, "Cloud AutoML")
 
    def __init__(self, target, service):
        self._target = target
        self._service = service
 
    @property
    def target(self):
        return self._target
 
    @property
    def service(self):
        return self._service
 
 
class UserSecretsClient():
    GET_USER_SECRET_ENDPOINT = '/requests/GetUserSecretRequest'
    GET_USER_SECRET_BY_LABEL_ENDPOINT = '/requests/GetUserSecretByLabelRequest'
    BIGQUERY_TARGET_VALUE = 1
 
    def __init__(self):
        url_base_override = os.getenv(_KAGGLE_URL_BASE_ENV_VAR_NAME)
        self.url_base = url_base_override or _KAGGLE_DEFAULT_URL_BASE
        # Follow the OAuth 2.0 Authorization standard (https://tools.ietf.org/html/rfc6750)
        self.jwt_token = os.getenv(_KAGGLE_USER_SECRETS_TOKEN_ENV_VAR_NAME)
        if self.jwt_token is None:
            raise CredentialError(
                'A JWT Token is required to use the UserSecretsClient, '
                f'but none found in environment variable {_KAGGLE_USER_SECRETS_TOKEN_ENV_VAR_NAME}')
        self.headers = {'Content-type': 'application/json'}
 
    def _make_post_request(self, data: dict, endpoint: str = GET_USER_SECRET_ENDPOINT) -> dict:
        # TODO(b/148309982) This code and the code in the constructor should be
        # removed and this class should use the new KaggleWebClient class instead.
        url = f'{self.url_base}{endpoint}'
        request_body = dict(data)
        request_body['JWE'] = self.jwt_token
        req = urllib.request.Request(url, headers=self.headers, data=bytes(
            json.dumps(request_body), encoding="utf-8"))
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECS) as response:
                response_json = json.loads(response.read())
                if not response_json.get('wasSuccessful') or 'result' not in response_json:
                    raise BackendError(
                        f'Unexpected response from the service. Response: {response_json}.')
                return response_json['result']
        except (URLError, socket.timeout) as e:
            if isinstance(
                    e, socket.timeout) or isinstance(
                    e.reason, socket.timeout):
                raise ConnectionError(
                    'Timeout error trying to communicate with service. Please ensure internet is on.') from e
            raise ConnectionError(
                'Connection error trying to communicate with service.') from e
        except HTTPError as e:
            if e.code == 401 or e.code == 403:
                raise CredentialError(
                    f'Service responded with error code {e.code}.'
                    ' Please ensure you have access to the resource.') from e
            raise BackendError('Unexpected response from the service.') from e
 
    def get_secret(self, label) -> str:
        """Retrieves a user secret value by its label.
 
        This returns the value of the secret with the given label,
        if it attached to the current kernel.
        Example usage:
            client = UserSecretsClient()
            secret = client.get_secret('my_db_password')
        """
        if label is None or len(label) == 0:
            raise ValidationError("Label must be non-empty.")
        request_body = {
            'Label': label,
        }
        response_json = self._make_post_request(request_body, self.GET_USER_SECRET_BY_LABEL_ENDPOINT)
        if 'secret' not in response_json:
            raise BackendError(
                f'Unexpected response from the service. Response: {response_json}')
        return response_json['secret']
 
    def get_bigquery_access_token(self) -> Tuple[str, Optional[datetime]]:
        """Retrieves BigQuery access token information from the UserSecrets service.
 
        This returns the token for the current kernel as well as its expiry (abs time) if it
        is present.
        Example usage:
            client = UserSecretsClient()
            token, expiry = client.get_bigquery_access_token()
        """
        return self._get_access_token(GcpTarget.BIGQUERY)
 
    def _get_gcs_access_token(self) -> Tuple[str, Optional[datetime]]:
        return self._get_access_token(GcpTarget.GCS)
 
    def _get_automl_access_token(self) -> Tuple[str, Optional[datetime]]:
        return self._get_access_token(GcpTarget.AUTOML)
 
    def _get_access_token(self, target: GcpTarget) -> Tuple[str, Optional[datetime]]:
        request_body = {
            'Target': target.target
        }
        response_json = self._make_post_request(request_body)
        if 'secret' not in response_json:
            raise BackendError(
                f'Unexpected response from the service. Response: {response_json}')
        # Optionally return expiry if it is set.
        expiresInSeconds = response_json.get('expiresInSeconds')
        expiry = datetime.utcnow() + timedelta(seconds=expiresInSeconds) if expiresInSeconds else None
        return response_json['secret'], expiry
 

{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 31231,
      "digest": "sha256:79f52292b1d0c079b84bc79d002947be409a09b15a1320355a4de834f57b2ee8"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 45339314,
         "digest": "sha256:c5e155d5a1d130a7f8a3e24cee0d9e1349bff13f90ec6a941478e558fde53c14"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 95104141,
         "digest": "sha256:86534c0d13b7196a49d52a65548f524b744d48ccaf89454659637bee4811d312"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1571501372,
         "digest": "sha256:5764e90b1fae3f6050c1b56958da5e94c0d0c2a5211955f579958fcbe6a679fd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1083072,
         "digest": "sha256:ba67f7304613606a1d577e2fc5b1e6bb14b764bcc8d07021779173bcc6a8d4b6"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 526,
         "digest": "sha256:19abed793cf0a9952e1a08188dbe2627ed25836757d0e0e3150d5c8328562b4e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 458,
         "digest": "sha256:df204f1f292ae58e4c4141a950fad3aa190d87ed9cc3d364ca6aa1e7e0b73e45"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 13119161,
         "digest": "sha256:1f7809135d9076fb9ed8ee186e93e3352c861489e0e80804f79b2b5634b456dd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 555884253,
         "digest": "sha256:03a365d6218dbe33f5b17d305f5e25e412f7b83b38394c5818bde053a542f11b"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 102870915,
         "digest": "sha256:00e3d0b7af78551716541d2076836df5594948d5d98f04f382158ef26eb7c907"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 95925388,
         "digest": "sha256:59782fefadba835c1e83cecdd73dc8e81121eae05ba58d3628a44a1c607feb6e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 142481172,
         "digest": "sha256:f81b01cf2c3f02e153a71704cc5ffe6102757fb7c2fcafc107a64581b0f6dc10"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1128076783,
         "digest": "sha256:f08bbb5c2bce948f0d12eea15c88aad45cdd5b804b71bee5a2cfdbf53c7ec254"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 444800302,
         "digest": "sha256:b831800c60a36c21033cb6e85f0bd3a5f5c9d96b2fa2797d0e8d4c50598180b8"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 157365696,
         "digest": "sha256:6d354ec67fa4ccf30460efadef27d48edf9599348cbab789c388f1d3a7fee232"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 63237273,
         "digest": "sha256:464f9b4eca5cdf36756cf0bef8c76b23344d0e975667becb743e8d6b9019c3cd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 427820542,
         "digest": "sha256:6c1f6bcbc63b982a86dc94301c2996505cec571938c60a434b3de196498c7b89"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 44581800,
         "digest": "sha256:c0a8110c6fede3cf54fa00a1b4e2fcb136d00b3cf490f658ec6d596e313c986e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 127637178,
         "digest": "sha256:c25df885c8dea40780f5b479bb6c7be924763399a21fa46c204d5bfac45056bd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 956429221,
         "digest": "sha256:7c1d98590e22f78a1b820f89b6ce245321437639957264e628b4abf4862e1223"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 586276809,
         "digest": "sha256:aab720d802b7d006b679ac10df4a259b3556812dea4dfc52d6111db47fc41e62"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 21717560,
         "digest": "sha256:5ee4a4cda8613a3fb172a827143aadacb98128479a22a280495604f989bf4483"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 93512644,
         "digest": "sha256:c4699852e987bc3fe9adde2544ffa690ad52ebec229c20f7e4153b015ac238ff"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 19141,
         "digest": "sha256:8d93692c8dcecacb8aca746a868f53d0b0cf1207e08ced8ffb2134bb01c1f871"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 84125618,
         "digest": "sha256:57c74d175611802a57531be97d19f586dc9cd810a5490eab04fd40b648312ead"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 3261,
         "digest": "sha256:1ac7a265bf03308e06e9cad7e77d12b22ca8bc6b7791d46398d95977e0042574"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2162,
         "digest": "sha256:1b4a5be69a4439f3de72877b7d408400e3aa0b4c37e9c70c4490b480bce682c0"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1270,
         "digest": "sha256:648046d6f6c28a42a39c9e68a9da90ccdabbd1ecfd0be77941114df4eb2406a4"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 644,
         "digest": "sha256:19a794f6956d460edfe74d5562d44366a7cf8bd46d83f408f1bf3c46e7282464"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2052,
         "digest": "sha256:880f92e310c2e03c28c5db85b342069b1a56cd13de7998ae52f699829774f075"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 875,
         "digest": "sha256:cad389727d6cd1696ed7e91b70eedd4c86fd30babb648e7be6cc1639582b0928"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 214,
         "digest": "sha256:c873da9a657a590abeae80bd3c0d0d87a6bfdfaf1d3873a0f210760a4050d6db"
      }
   ]
}

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.signal import butter, lfilter
from scipy.stats import norm
from sklearn import base
from statsmodels.distributions.empirical_distribution import ECDF
import logging
import numpy as np


logger = logging.getLogger('Kaggler')


class QuantileEncoder(base.BaseEstimator):
    """QuantileEncoder encodes numerical features to quantile values.

    Attributes:
        ecdfs (list of empirical CDF): empirical CDFs for columns
        n_label (int): the number of labels to be created.
    """

    def __init__(self, n_label=10, sample=100000, random_state=42):
        """Initialize a QuantileEncoder class object.

        Args:
            n_label (int): the number of labels to be created.
            sample (int or float): the number or fraction of samples for ECDF
        """
        self.n_label = n_label
        self.sample = sample
        self.random_state = random_state

    def fit(self, X, y=None):
        """Get empirical CDFs of numerical features.

        Args:
            X (pandas.DataFrame): numerical features to encode

        Returns:
            A trained QuantileEncoder object.
        """
        def _calculate_ecdf(x):
            return ECDF(x[~np.isnan(x)])

        if self.sample >= X.shape[0]:
            self.ecdfs = X.apply(_calculate_ecdf, axis=0)
        elif self.sample > 1:
            self.ecdfs = X.sample(n=self.sample,
                                  random_state=self.random_state).apply(
                                      _calculate_ecdf, axis=0
                                  )
        else:
            self.ecdfs = X.sample(frac=self.sample,
                                  random_state=self.random_state).apply(
                                      _calculate_ecdf, axis=0
                                  )

        return self

    def fit_transform(self, X, y=None):
        """Get empirical CDFs of numerical features and encode to quantiles.

        Args:
            X (pandas.DataFrame): numerical features to encode

        Returns:
            Encoded features (pandas.DataFrame).
        """
        self.fit(X, y)

        return self.transform(X)

    def transform(self, X):
        """Encode numerical features to quantiles.

        Args:
            X (pandas.DataFrame): numerical features to encode

        Returns:
            Encoded features (pandas.DataFrame).
        """
        for i, col in enumerate(X.columns):
            X.loc[:, col] = self._transform_col(X[col], i)

        return X

    def _transform_col(self, x, i):
        """Encode one numerical feature column to quantiles.

        Args:
            x (pandas.Series): numerical feature column to encode
            i (int): column index of the numerical feature

        Returns:
            Encoded feature (pandas.Series).
        """
        # Map values to the emperical CDF between .1% and 99.9%
        rv = np.ones_like(x) * -1

        filt = ~np.isnan(x)
        rv[filt] = np.floor((self.ecdfs[i](x[filt]) * 0.998 + .001) *
                            self.n_label)

        return rv


class Normalizer(base.BaseEstimator):
    """Normalizer that transforms numerical columns into normal distribution.

    Attributes:
        ecdfs (list of empirical CDF): empirical CDFs for columns
    """

    def fit(self, X, y=None):
        self.ecdfs = [None] * X.shape[1]

        for col in range(X.shape[1]):
            self.ecdfs[col] = ECDF(X[col].values)

        return self

    def transform(self, X):
        """Normalize numerical columns.

        Args:
            X (pandas.DataFrame) : numerical columns to normalize

        Returns:
            (pandas.DataFrame): normalized numerical columns
        """

        for col in range(X.shape[1]):
            X[col] = self._transform_col(X[col], col)

        return X

    def fit_transform(self, X, y=None):
        """Normalize numerical columns.

        Args:
            X (pandas.DataFrame) : numerical columns to normalize

        Returns:
            (pandas.DataFrame): normalized numerical columns
        """

        self.ecdfs = [None] * X.shape[1]

        for col in range(X.shape[1]):
            self.ecdfs[col] = ECDF(X[col].values)
            X[col] = self._transform_col(X[col], col)

        return X

    def _transform_col(self, x, col):
        """Normalize one numerical column.

        Args:
            x (pandas.Series): a numerical column to normalize
            col (int): column index

        Returns:
            A normalized feature vector.
        """

        return norm.ppf(self.ecdfs[col](x.values) * .998 + .001)


class BandpassFilter(base.BaseEstimator):

    def __init__(self, fs=10., lowcut=.5, highcut=3., order=3):
        self.fs = 10.
        self.lowcut = .5
        self.highcut = 3.
        self.order = 3
        self.b, self.a = self._butter_bandpass()

    def _butter_bandpass(self):
        nyq = .5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')

        return b, a

    def _butter_bandpass_filter(self, x):
        return lfilter(self.b, self.a, x)

    def fit(self, X):
        return self

    def transform(self, X, y=None):
        for col in range(X.shape[1]):
            X[:, col] = self._butter_bandpass_filter(X[:, col])

        return X
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

nRowsRead = 1000 # specify 'None' if want to read whole file
# us-counties.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/us-counties.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'us-counties.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 9, 10)
nRowsRead = 1000 # specify 'None' if want to read whole file
# us-states.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv('/kaggle/input/us-states.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'us-states.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)
plotCorrelationMatrix(df2, 8)
plotScatterMatrix(df2, 9, 10)