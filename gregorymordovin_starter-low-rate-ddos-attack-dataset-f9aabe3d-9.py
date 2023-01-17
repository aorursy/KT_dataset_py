from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))
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
    filename = 'df'#df.dataframeName
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

"""
Reading the data
https://jon.oberheide.org/blog/2008/10/15/dpkt-tutorial-2-parsing-a-pcap-file/
https://dpkt.readthedocs.io/en/latest/_modules/examples/print_packets.html
"""

import datetime
import socket

import dpkt
import numpy as np
import pandas as pd
from dpkt.compat import compat_ord


def mac_addr(address):
    """Convert a MAC address to a readable/printable string

       Args:
           address (str): a MAC address in hex form (e.g. '\x01\x02\x03\x04\x05\x06')
       Returns:
           str: Printable/readable MAC address
    """
    return ':'.join('%02x' % compat_ord(b) for b in address)


def inet_to_str(inet):
    """Convert inet object to a string

        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


##################################################################################

packets_to_read = 10 ** 10
f = open('../input/ICMP.pcap', 'rb')
pcap = dpkt.pcapng.Reader(f)
dpcap = {
    'ts': [],
    'eth_src': [],
    'eth_dst': [],
    'eth_type': [],
    'ip_src': [],
    'ip_dst': [],
    'ip_len': [],
    'ip_ttl': [],
    'ip_df': [],
    'ip_mf': [],
    'ip_offset': [],
    'type': [],
    'tcp_dport': [],
    'http_uri': [],
    'http_method': [],
    'http_version': [],
    'http_headers_ua': [],
}
non_ip_packets = dict()
c = 0
# For each packet in the pcap process the contents
for timestamp, buf in pcap:
    
    c += 1
    if c > packets_to_read:
        break

    dpcap['ts'].append(str(datetime.datetime.utcfromtimestamp(timestamp)))  # timestamp in UTC

    # Unpack the Ethernet frame (mac src/dst, ethertype)
    eth = dpkt.ethernet.Ethernet(buf)
    dpcap['eth_src'].append(mac_addr(eth.src))
    dpcap['eth_dst'].append(mac_addr(eth.dst))
    dpcap['eth_type'].append(eth.type)

    # Make sure the Ethernet frame contains an IP packet
    if not isinstance(eth.data, dpkt.ip.IP):
        #print('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
        non_ip_packets[eth.data.__class__.__name__] = non_ip_packets.get(eth.data.__class__.__name__, 0) + 1
        
        dpcap['ts'].pop()
        dpcap['eth_src'].pop()
        dpcap['eth_dst'].pop()
        dpcap['eth_type'].pop()
        
        continue

    # Now unpack the data within the Ethernet frame (the IP packet)
    # Pulling out src, dst, length, fragment info, TTL, and Protocol
    ip = eth.data

    # Pull out fragment information (flags and offset all packed into off field, so use bitmasks)
    do_not_fragment = bool(ip.off & dpkt.ip.IP_DF)
    more_fragments = bool(ip.off & dpkt.ip.IP_MF)
    fragment_offset = ip.off & dpkt.ip.IP_OFFMASK

    dpcap['ip_src'].append(inet_to_str(ip.src))
    dpcap['ip_dst'].append(inet_to_str(ip.dst))
    dpcap['ip_len'].append(ip.len)
    dpcap['ip_ttl'].append(ip.ttl)
    dpcap['ip_df'].append(do_not_fragment)
    dpcap['ip_mf'].append(more_fragments)
    dpcap['ip_offset'].append(fragment_offset)

    dpcap['type'].append(ip.data.__class__.__name__)
    
    dpcap['tcp_dport'].append(np.nan)
    dpcap['http_uri'].append(np.nan)
    dpcap['http_method'].append(np.nan)
    dpcap['http_version'].append(np.nan)
    dpcap['http_headers_ua'].append(np.nan)
    
    if not isinstance(ip.data, dpkt.tcp.TCP):
        continue
    
    tcp = ip.data
    
    dpcap['tcp_dport'].pop()
    dpcap['tcp_dport'].append(tcp.dport)

    if tcp.dport == 80 and len(tcp.data) > 2 or True:
        #print(tcp.data)
        try:
            http = dpkt.http.Request(tcp.data)
            dpcap['http_uri'].pop()
            dpcap['http_uri'].append(http.uri)
            #print http.uri
            dpcap['http_method'][-1] = http.method
            dpcap['http_version'][-1] = http.version
            dpcap['http_headers_ua'][-1] = http.headers['user-agent']
        except:
            pass

df = pd.DataFrame.from_dict(dpcap)
print('non_ip_packets counter:', non_ip_packets)
df[df.http_uri.notnull()].head(20)
#df.astype('object').describe().transpose()
df.groupby(df.type).nunique()

#plotPerColumnDistribution(df, nGraphShown=50, nGraphPerRow=3)
import collections

collections.Counter(df.http_uri)

df[['ip_src', 'ip_dst', 'http_headers_ua', 'http_uri', 'ts']].groupby(['ip_dst', 'ip_src', ]).agg(['count', 'min', 'max'])